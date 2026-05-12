"""Lagrangian SAC for ~/bimanual_peghole.

约束化 SAC: 把 collision 等安全约束信号 (env._create_info_dictionary["cost"])
从 reward 里剥出来, 用 cost critic Q_C 单独学; actor 目标变为
    max E[ Q_R - λ·Q_C - α·log π ]
λ 通过 dual ascent 学到约束 d (per-step cost_limit) 满足:
    log_λ ← log_λ + lr_λ · ((Q_C·(1-γ_c)) - cost_limit_per_step)

数据流 (probe_extra_info_cost.py 已验证):
    env._create_info_dictionary 返回 {"cost": tensor}
    → IsaacSim.step_all extra_info → VectorCore → VectorizedDataset.append_vectorized
    → Dataset.flatten() / ExtraInfo.flatten()
    → fit_dataset.info.data["cost"] 与 reward 同形 1D, 顺序对齐.

文件位置约束: pickle 路径绑定 algorithm.lagrangian_sac.SACLagrangian. 任何 load
checkpoint 的脚本必须 sys.path 包含 PROJECT_ROOT (train_sac.py:33 同款).
不要重命名 / 移动文件, 否则旧 checkpoint 失效.

双 critic 取舍:
    reward critic: min — 标准 SAC, 防 Q 过估;
    cost   critic: max — 保守高估 cost, 不容易低估安全风险 (WCSAC 风格).

cost critic 的 target 不带 entropy 项 (cost 是普通 Bellman, 不是 soft 目标).
"""

import math
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim

from mushroom_rl.algorithms.actor_critic import SAC
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.core import Serializable, Dataset, DatasetInfo
from mushroom_rl.utils.torch import TorchUtils


class ConstrainedReplayMemory(Serializable):
    """单 buffer 存 (s, a, r, c, s', absorb, last). 照搬 mushroom ReplayMemory 的
    ring 逻辑, 多一列 cost. 不做 n-step return (SAC 默认 1-step).

    cost 通过 dataset.info.data["cost"] 取 — flatten 后 1D 与 reward 同形对齐.
    """

    def __init__(self, mdp_info, agent_info, initial_size, max_size):
        self._initial_size = initial_size
        self._max_size = max_size
        self._idx = 0
        self._full = False
        self._mdp_info = mdp_info
        self._agent_info = agent_info

        assert agent_info.backend in ("numpy", "torch"), (
            f"backend {agent_info.backend} 不支持; 需 numpy 或 torch"
        )

        self._dataset = None
        self._costs = None
        self.reset()

        self._add_save_attr(
            _initial_size='primitive',
            _max_size='primitive',
            _mdp_info='mushroom',
            _agent_info='mushroom',
            _idx='primitive!',
            _full='primitive!',
            _dataset='mushroom!',
            _costs='torch!',
        )

    def reset(self):
        self._idx = 0
        self._full = False
        dataset_info = DatasetInfo.create_replay_memory_info(self._mdp_info, self._agent_info)
        self._dataset = Dataset(dataset_info, n_steps=self._max_size)
        if self._agent_info.backend == "torch":
            device = TorchUtils.get_device(None)
            self._costs = torch.empty(self._max_size, dtype=torch.float, device=device)
        else:
            self._costs = np.empty(self._max_size, dtype=float)

    def add(self, dataset):
        s, a, r, sp, absorb, last = dataset.parse(to=self._agent_info.backend)
        if "cost" not in dataset.info.data:
            raise KeyError(
                "ConstrainedReplayMemory: dataset.info.data 没有 'cost' 键. "
                "确保 env._create_info_dictionary 返回 {'cost': tensor}."
            )
        cost = dataset.info.data["cost"]
        if self._agent_info.backend == "torch":
            cost = torch.as_tensor(cost, dtype=torch.float, device=self._costs.device)
        else:
            cost = np.asarray(cost, dtype=float)

        n = len(dataset)
        for i in range(n):
            if self._full:
                self._dataset.state[self._idx] = s[i]
                self._dataset.action[self._idx] = a[i]
                self._dataset.reward[self._idx] = r[i]
                self._dataset.next_state[self._idx] = sp[i]
                self._dataset.absorbing[self._idx] = absorb[i]
                self._dataset.last[self._idx] = last[i]
            else:
                sample = [s[i], a[i], r[i], sp[i], absorb[i], last[i]]
                self._dataset.append(sample, {})

            self._costs[self._idx] = cost[i]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        idxs = self._dataset.array_backend.randint(0, len(self._dataset), (n_samples,))
        batch = self._dataset[idxs]
        s, a, r, sp, absorb, last = batch.parse()
        c = self._costs[idxs]
        return s, a, r, c, sp, absorb, last

    @property
    def initialized(self):
        return self.size > self._initial_size

    @property
    def size(self):
        return self._idx if not self._full else self._max_size

    def _post_load(self):
        if self._full is None:
            self.reset()


class SACLagrangian(SAC):
    """SAC + per-step cost constraint via Lagrange multiplier.

    actor loss: (α·logπ - Q_R + λ·Q_C).mean()
    cost critic: 普通 Bellman, q_c_target = c + γ_c · max_θ' Q_C^target(s', π(s'))
    λ 更新: log_λ += lr_λ · violation, where violation = Q_C·(1-γ_c) - cost_limit
            (Q_C 是 discounted sum, 乘 (1-γ_c) 还原到 per-step 量纲)

    cost critic 在 critic warmup 期间也学 (跟 reward critic 同步), actor / α / λ
    在 warmup 后才更新.
    """

    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer,
                 critic_params, batch_size, initial_replay_size, max_replay_size,
                 warmup_transitions, tau, lr_alpha,
                 cost_limit, lr_lambda,
                 lambda_max=100.0, init_log_lambda=0.0,
                 cost_critic_params=None, gamma_cost=None,
                 use_log_alpha_loss=False, log_std_min=-20, log_std_max=2,
                 target_entropy=None, critic_fit_params=None):
        """
        Args:
            cost_limit (float): per-step cost 预算 (e.g. 0.01 = 容忍 1% collision rate).
            lr_lambda (float): Lagrange 乘子学习率. 通常 1e-3 ~ 1e-4, 比 lr_actor 低.
            lambda_max (float): λ clamp 上限 (默认 100).
            init_log_lambda (float): log_λ 初值 (默认 0 → λ_init=1).
            cost_critic_params (dict, None): cost critic 网络配置. None 则复用
                critic_params (相同结构两套权重).
            gamma_cost (float, None): cost MDP 折扣. None 则复用 mdp_info.gamma.
                设 1.0 对应 average-cost (注意此时 cost_limit 不能除 (1-γ_c), 见
                _update_lambda).
            其余参数同 SAC.
        """
        super().__init__(
            mdp_info=mdp_info,
            actor_mu_params=actor_mu_params,
            actor_sigma_params=actor_sigma_params,
            actor_optimizer=actor_optimizer,
            critic_params=critic_params,
            batch_size=batch_size,
            initial_replay_size=initial_replay_size,
            max_replay_size=max_replay_size,
            warmup_transitions=warmup_transitions,
            tau=tau,
            lr_alpha=lr_alpha,
            use_log_alpha_loss=use_log_alpha_loss,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            target_entropy=target_entropy,
            critic_fit_params=critic_fit_params,
        )

        # 替换 reward-only replay → constrained replay.
        # 父类已用 mushroom 的 ReplayMemory 占了一份 (max_replay_size 的小预分配),
        # 这里直接覆盖, 旧 buffer 让 GC 回收.
        self._replay_memory = ConstrainedReplayMemory(
            mdp_info, self.info, initial_replay_size, max_replay_size
        )

        # cost critic: 默认复用 reward critic 的 params (网络结构/优化器一样, 权重独立).
        if cost_critic_params is None:
            cost_critic_params = critic_params
        c_params = deepcopy(cost_critic_params)
        c_target_params = deepcopy(cost_critic_params)
        if 'n_models' in c_params:
            assert c_params['n_models'] == 2
        else:
            c_params['n_models'] = 2
            c_target_params['n_models'] = 2

        self._cost_critic_approximator = Regressor(TorchApproximator, **c_params)
        self._target_cost_critic_approximator = Regressor(TorchApproximator, **c_target_params)
        self._init_target(self._cost_critic_approximator,
                          self._target_cost_critic_approximator)

        self._log_lambda = torch.tensor(float(init_log_lambda),
                                        dtype=torch.float32, requires_grad=True)
        self._lambda_optim = optim.Adam([self._log_lambda], lr=lr_lambda)

        self._cost_limit = float(cost_limit)
        self._lambda_max = float(lambda_max)
        if gamma_cost is None:
            self._gamma_cost = float(mdp_info.gamma)
        else:
            self._gamma_cost = float(gamma_cost)

        self._add_save_attr(
            _cost_critic_approximator='mushroom',
            _target_cost_critic_approximator='mushroom',
            _log_lambda='torch',
            _lambda_optim='torch',
            _cost_limit='primitive',
            _lambda_max='primitive',
            _gamma_cost='primitive',
        )

    def fit(self, dataset):
        if "cost" in dataset.info.data:
            self._replay_memory.add(dataset)
        if not self._replay_memory.initialized:
            return
        s, a, r, c, sp, absorb, last = self._replay_memory.get(self._batch_size())

        # actor / α / λ 仅在 critic warmup 后更新; cost critic & reward critic 总是更新.
        if self._replay_memory.size > self._warmup_transitions():
            a_new, log_prob = self.policy.compute_action_and_log_prob_t(s)
            loss = self._loss(s, a_new, log_prob)
            self._optimize_actor_parameters(loss)
            self._update_alpha(log_prob.detach())
            self._update_lambda(s)

        # reward critic: 同 SAC.
        q_next = self._next_q(sp, absorb)
        q_target = r + self.mdp_info.gamma * q_next
        self._critic_approximator.fit(s, a, q_target, **self._critic_fit_params)
        self._update_target(self._critic_approximator, self._target_critic_approximator)

        # cost critic: 普通 Bellman, 不带 entropy 项.
        qc_next = self._next_qc(sp, absorb)
        qc_target = c + self._gamma_cost * qc_next
        self._cost_critic_approximator.fit(s, a, qc_target, **self._critic_fit_params)
        self._update_target(self._cost_critic_approximator,
                            self._target_cost_critic_approximator)

    def _loss(self, state, action_new, log_prob):
        q_r = torch.min(
            self._critic_approximator(state, action_new, idx=0),
            self._critic_approximator(state, action_new, idx=1),
        )
        q_c = torch.max(
            self._cost_critic_approximator(state, action_new, idx=0),
            self._cost_critic_approximator(state, action_new, idx=1),
        )
        lam = self._log_lambda.exp().detach()
        return (self._alpha * log_prob - q_r + lam * q_c).mean()

    def _next_qc(self, next_state, absorbing):
        a, _ = self.policy.compute_action_and_log_prob(next_state)
        qc = self._target_cost_critic_approximator.predict(next_state, a, prediction='max')
        qc *= 1 - absorbing.to(int)
        return qc

    def _update_lambda(self, state):
        with torch.no_grad():
            a, _ = self.policy.compute_action_and_log_prob(state)
            q_c = torch.max(
                self._cost_critic_approximator(state, a, idx=0),
                self._cost_critic_approximator(state, a, idx=1),
            )
            # Q_C ≈ Σ γ_c^t c_t, per-step cost ≈ Q_C·(1-γ_c). γ_c=1 退化为 average-cost,
            # 此时 cost_limit 直接是 Q_C 量纲, 不缩放.
            if self._gamma_cost < 1.0:
                per_step_cost = q_c.mean() * (1.0 - self._gamma_cost)
            else:
                per_step_cost = q_c.mean()
            violation = (per_step_cost - self._cost_limit).detach()

        loss_lambda = -(self._log_lambda * violation)
        self._lambda_optim.zero_grad()
        loss_lambda.backward()
        self._lambda_optim.step()

        with torch.no_grad():
            self._log_lambda.clamp_(min=-10.0, max=math.log(self._lambda_max))

    def _post_load(self):
        super()._post_load()
        self._update_lambda_optimizer_parameters()

    def _update_lambda_optimizer_parameters(self):
        if self._lambda_optim is not None:
            TorchUtils.update_optimizer_parameters(self._lambda_optim, [self._log_lambda])

    @property
    def _lambda(self):
        return self._log_lambda.exp()
