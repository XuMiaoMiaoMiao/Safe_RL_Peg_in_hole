"""Lagrangian SAC for ~/bimanual_peghole.

约束化 SAC: 把 collision 等安全约束信号 (env._create_info_dictionary["cost"])
从 reward 里剥出来, 用 cost critic Q_C 单独学; actor 目标变为
    max E[ Q_R - λ·Q_C - α·log π ]
λ 默认通过 latest env batch 与 replay batch 的较大 per-step cost rate 做 dual ascent:
    log_λ ← log_λ + lr_λ · (max(mean(c_recent), mean(c_replay)) - cost_limit_per_step)
episode_rate 模式 (eval-based, 有滞后): λ 只接受来自 eval 的完整 episode 统计
    log_λ ← log_λ + lr_λ · (cost_sum / n_episodes - cost_limit_per_episode)
    不使用 replay random batch, 不依赖 Mushroom flattened dataset 的 last 标志;
    但 eval 在整个 epoch 之后才运行, 且使用 deterministic policy, 与训练 policy 存在偏差.
rollout_episode_rate 模式 (推荐): λ 使用当前 policy rollout 中完成的真实 episode 统计
    log_λ ← log_λ + lr_λ · (mean(episode_cost_sums) - cost_limit_per_episode)
    信号来自 EpisodeCostTracker (train_sac_lagrangian.py 中的 callback_step 钩子),
    逐 vector-step 实时累计 cost, episode 完成时推入滑动窗口, epoch 结束后 drain 更新 λ.
    避免了 episode_rate 的 epoch 级滞后与 deterministic policy 偏差.
也可用旧版 discounted_q 模式:
    log_λ ← log_λ + lr_λ · ((Q_C·(1−γ_c)) - cost_limit_per_step)

数据流 (已验证):
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
    """SAC + cost constraint via Lagrange multiplier.

    actor loss: (α·logπ - Q_R + λ·Q_C).mean()
    cost critic: 普通 Bellman, q_c_target = c + γ_c · max_θ' Q_C^target(s', π(s'))
    λ 更新默认用 max_recent_replay:
        violation = max(mean(c_recent), mean(c_replay)) - cost_limit
    batch_cost_rate / recent_cost_rate 可作为消融模式.
    episode_rate 不在 fit() 里更新; 必须由训练脚本用完整 episode 统计显式调用
    update_lambda_from_episode_statistics(). 这样不会把 replay batch 的 last.sum()
    或 VectorizedDataset.flatten() 注入的 last=True 当成 episode 数。
    旧版 discounted_q 可通过 lambda_update_mode="discounted_q" 启用:
        violation = Q_C·(1-γ_c) - cost_limit
        (Q_C 是 discounted sum, 乘 (1-γ_c) 还原到 per-step 量纲)

    cost critic 在 critic warmup 期间也学 (跟 reward critic 同步), actor / α / λ
    在 warmup 后才更新.
    """

    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer,
                 critic_params, batch_size, initial_replay_size, max_replay_size,
                 warmup_transitions, tau, lr_alpha,
                 cost_limit, lr_lambda,
                 lambda_max=100.0, lambda_min=0.0, init_log_lambda=0.0,
                 cost_critic_params=None, gamma_cost=None,
                 lambda_update_mode="max_recent_replay",
                 actor_grad_clip=None,
                 use_log_alpha_loss=False, log_std_min=-20, log_std_max=2,
                 target_entropy=None, critic_fit_params=None):
        """
        Args:
            cost_limit (float): cost 预算. per-step 模式下是每步 cost rate;
                episode_rate 下是每集平均 cost sum.
            lr_lambda (float): Lagrange 乘子学习率. 通常 1e-3 ~ 1e-4, 比 lr_actor 低.
            lambda_max (float): λ clamp 上限 (默认 100).
            lambda_min (float): λ clamp 下限 (默认 0). 设 >0 防止约束信号完全消失,
                warmstart 场景建议 0.05~0.1.
            init_log_lambda (float): log_λ 初值 (默认 0 → λ_init=1).
            cost_critic_params (dict, None): cost critic 网络配置. None 则复用
                critic_params (相同结构两套权重).
            gamma_cost (float, None): cost MDP 折扣. None 则复用 mdp_info.gamma.
                设 1.0 对应 average-cost (注意此时 cost_limit 不能除 (1-γ_c), 见
                _update_lambda).
            lambda_update_mode (str): λ 更新信号来源.
                "rollout_episode_rate" (推荐): cost_limit = 每集平均碰撞次数.
                    fit() 不更新 λ; 训练脚本在每个 epoch 结束后调用
                    update_lambda_from_rollout_episodes() (由 EpisodeCostTracker 提供数据).
                    使用当前 policy 的 rollout 数据, 避免 eval-only 的滞后与偏差.
                "episode_rate" (eval-based, 有滞后): cost_limit = 每集平均碰撞次数.
                    fit() 不更新 λ; 必须由训练脚本在 eval 后显式调用
                    update_lambda_from_episode_statistics(). 使用 deterministic policy
                    的 eval 数据, 与训练 policy 存在一个 epoch 的滞后.
                "max_recent_replay": 用最新 env batch 与 replay batch per-step cost
                    rate 的较大值, 在 fit() 中更新.
                "batch_cost_rate": 仅用 replay batch per-step rate, 在 fit() 中更新.
                "recent_cost_rate": 仅用最新 env batch per-step rate, 在 fit() 中更新.
                "discounted_q": Q_C×(1-gamma_cost) 缩放到 per-step 量纲, 在 fit() 中更新.
            actor_grad_clip (float, None): actor 梯度 L2 norm 上限. None = 不裁剪.
                warmstart 后 critic warmup 结束时第一次 actor 更新容易梯度爆炸, 建议 1.0.
            其余参数同 SAC.
        """
        valid_lambda_modes = (
            "max_recent_replay", "batch_cost_rate", "recent_cost_rate", "discounted_q",
            "episode_rate", "rollout_episode_rate",
        )
        if lambda_update_mode not in valid_lambda_modes:
            raise ValueError(
                "lambda_update_mode 必须是 'max_recent_replay', 'batch_cost_rate', "
                "'recent_cost_rate', 'discounted_q', 'episode_rate' 或 "
                f"'rollout_episode_rate', 当前 {lambda_update_mode!r}"
            )

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
        # λ 是单个 dual scalar; Adam 的 momentum 会在 violation 变负后仍继续
        # 推高 λ, 让日志出现 "eval_violation<0 但 λ 还涨" 的反直觉行为.
        # 用无 momentum 的 SGD 保持 log_λ ← log_λ + lr·violation 的直接语义.
        self._lambda_optim = optim.SGD([self._log_lambda], lr=lr_lambda)

        self._cost_limit = float(cost_limit)
        self._lambda_max = float(lambda_max)
        self._lambda_min = float(lambda_min)
        self._lambda_update_mode = str(lambda_update_mode)
        self._lambda_qc_mean = float("nan")
        self._lambda_batch_cost_rate = float("nan")
        self._lambda_recent_cost_rate = float("nan")
        self._lambda_selected_cost_rate = float("nan")
        self._lambda_internal_violation = float("nan")
        self._lambda_update_source = "none"
        self._actor_grad_clip = float(actor_grad_clip) if actor_grad_clip is not None else None
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
            _lambda_min='primitive',
            _lambda_update_mode='primitive',
            _lambda_qc_mean='primitive',
            _lambda_batch_cost_rate='primitive',
            _lambda_recent_cost_rate='primitive',
            _lambda_selected_cost_rate='primitive',
            _lambda_internal_violation='primitive',
            _lambda_update_source='primitive',
            _actor_grad_clip='primitive',
            _gamma_cost='primitive',
        )

    def fit(self, dataset):
        recent_cost_rate = None
        if "cost" in dataset.info.data:
            recent_cost = torch.as_tensor(
                dataset.info.data["cost"], dtype=torch.float, device=self._log_lambda.device
            )
            # This is a per-step statistic only. Never infer episode counts from
            # dataset.last: VectorizedDataset.flatten() marks fit chunk boundaries
            # as last=True, so last.sum() is not a reliable episode count.
            recent_cost_rate = recent_cost.mean()
            self._replay_memory.add(dataset)
        if not self._replay_memory.initialized:
            return
        s, a, r, c, sp, absorb, last = self._replay_memory.get(self._batch_size())

        # actor / α 仅在 critic warmup 后更新; cost critic & reward critic 总是更新.
        # λ: per-step modes update here. episode_rate is updated externally from
        # complete episode statistics after eval.
        if self._replay_memory.size > self._warmup_transitions():
            a_new, log_prob = self.policy.compute_action_and_log_prob_t(s)
            loss = self._loss(s, a_new, log_prob)
            self._optimize_actor_parameters(loss)
            self._update_alpha(log_prob.detach())
            # episode_rate: λ is updated externally after eval via
            #   update_lambda_from_episode_statistics().
            # rollout_episode_rate: λ is updated externally after core.learn() via
            #   update_lambda_from_rollout_episodes() (EpisodeCostTracker).
            # All other modes: update λ here from the current fit batch.
            if self._lambda_update_mode not in ("episode_rate", "rollout_episode_rate"):
                self._update_lambda_from_fit_batch(s, c, recent_cost_rate)

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

    @staticmethod
    def _scalar_to_float(value):
        if value is None:
            return float("nan")
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _apply_lambda_violation(
        self,
        violation,
        selected_cost_rate,
        *,
        q_c_mean=None,
        batch_cost_rate=None,
        recent_cost_rate=None,
        source="unknown",
    ):
        violation = torch.as_tensor(
            violation, dtype=torch.float32, device=self._log_lambda.device
        )
        selected_cost_rate = torch.as_tensor(
            selected_cost_rate, dtype=torch.float32, device=self._log_lambda.device
        )

        self._lambda_qc_mean = self._scalar_to_float(q_c_mean)
        self._lambda_batch_cost_rate = self._scalar_to_float(batch_cost_rate)
        self._lambda_recent_cost_rate = self._scalar_to_float(recent_cost_rate)
        self._lambda_selected_cost_rate = self._scalar_to_float(selected_cost_rate)
        self._lambda_internal_violation = self._scalar_to_float(violation)
        self._lambda_update_source = str(source)

        loss_lambda = -(self._log_lambda * violation.detach())
        self._lambda_optim.zero_grad()
        loss_lambda.backward()
        self._lambda_optim.step()

        with torch.no_grad():
            log_min = math.log(self._lambda_min) if self._lambda_min > 0 else -10.0
            self._log_lambda.clamp_(min=log_min, max=math.log(self._lambda_max))

        return True

    def _update_lambda_from_fit_batch(self, state, cost, recent_cost_rate=None):
        with torch.no_grad():
            a, _ = self.policy.compute_action_and_log_prob(state)
            q_c = torch.max(
                self._cost_critic_approximator(state, a, idx=0),
                self._cost_critic_approximator(state, a, idx=1),
            )
            q_c_mean = q_c.mean()
            cost_f = cost.float()
            batch_cost_rate = cost_f.mean()

            if recent_cost_rate is None:
                recent_rate = None
                recent_rate_for_log = torch.full_like(batch_cost_rate, float("nan"))
            else:
                recent_rate = recent_cost_rate.to(batch_cost_rate.device)
                recent_rate_for_log = recent_rate

            if self._lambda_update_mode == "batch_cost_rate":
                selected_cost_rate = batch_cost_rate
                violation = selected_cost_rate - self._cost_limit
                source = "replay_batch_per_step"
            elif self._lambda_update_mode == "recent_cost_rate":
                if recent_rate is None:
                    self._lambda_update_source = "recent_per_step_missing"
                    return False
                selected_cost_rate = recent_rate if recent_rate is not None else batch_cost_rate
                violation = selected_cost_rate - self._cost_limit
                source = "recent_rollout_per_step"
            elif self._lambda_update_mode == "max_recent_replay":
                if recent_rate is None:
                    selected_cost_rate = batch_cost_rate
                    source = "replay_batch_per_step"
                else:
                    selected_cost_rate = torch.maximum(batch_cost_rate, recent_rate)
                    source = "max_recent_replay_per_step"
                violation = selected_cost_rate - self._cost_limit
            else:
                # Q_C ≈ Σ γ_c^t c_t, per-step cost ≈ Q_C·(1-γ_c). γ_c=1 退化为
                # average-cost, 此时 cost_limit 直接是 Q_C 量纲, 不缩放.
                if self._gamma_cost < 1.0:
                    per_step_cost = q_c_mean * (1.0 - self._gamma_cost)
                else:
                    per_step_cost = q_c_mean
                violation = per_step_cost - self._cost_limit
                selected_cost_rate = per_step_cost
                source = "discounted_cost_critic"

        return self._apply_lambda_violation(
            violation,
            selected_cost_rate,
            q_c_mean=q_c_mean,
            batch_cost_rate=batch_cost_rate,
            recent_cost_rate=recent_rate_for_log,
            source=source,
        )

    def update_lambda_from_episode_statistics(
        self,
        *,
        cost_sum=None,
        n_episodes=None,
        cost_episode_rate=None,
        source="eval_episode_rate",
    ):
        """Update λ from complete episode-level cost statistics.

        This is the only valid update path for lambda_update_mode="episode_rate".
        It deliberately does not read replay batches or Dataset.last, because
        Mushroom's flattened last can mark fit chunk boundaries rather than true
        environment episode endings.
        """
        if self._lambda_update_mode != "episode_rate":
            raise RuntimeError(
                "update_lambda_from_episode_statistics() 只用于 "
                "lambda_update_mode='episode_rate'"
            )
        if self._replay_memory.size <= self._warmup_transitions():
            self._lambda_update_source = "episode_rate_waiting_warmup"
            return False

        if cost_episode_rate is None:
            if cost_sum is None or n_episodes is None:
                raise ValueError(
                    "必须传 cost_episode_rate, 或同时传 cost_sum 和 n_episodes"
                )
            if n_episodes <= 0:
                raise ValueError(f"n_episodes 必须 > 0, 当前 {n_episodes}")
            cost_episode_rate = float(cost_sum) / float(n_episodes)

        selected_cost_rate = torch.as_tensor(
            float(cost_episode_rate),
            dtype=torch.float32,
            device=self._log_lambda.device,
        )
        violation = selected_cost_rate - self._cost_limit

        return self._apply_lambda_violation(
            violation,
            selected_cost_rate,
            q_c_mean=None,
            batch_cost_rate=None,
            recent_cost_rate=selected_cost_rate,
            source=source,
        )

    def update_lambda_from_rollout_episodes(
        self,
        cost_episode_rate,
        n_episodes,
        source="rollout_episode_rate",
    ):
        """Update λ from completed episodes collected during the current policy's rollout.

        This is the only valid update path for lambda_update_mode="rollout_episode_rate".
        It is called once per training epoch from the training script, after core.learn()
        finishes, using statistics provided by EpisodeCostTracker.drain().

        Compared to update_lambda_from_episode_statistics() (eval-based):
          - Uses training-rollout episodes, not eval episodes → true on-policy signal.
          - Called right after rollout, not after a full eval run → no epoch-level lag.
          - Uses stochastic policy, matching what the critic/actor see.

        Episode boundaries are detected in EpisodeCostTracker via VectorCore's
        callback_step `last` flag (= absorbing | timeout). This is the raw per-env
        signal from the env, NOT the Mushroom-flattened `last` (which sets the last
        step of every fit chunk to True regardless of episode endings — see
        VectorizedDataset.flatten() line: last_padded[-1, :] = True).

        Args:
            cost_episode_rate: mean total cost per completed episode (float).
                Computed as mean(episode_cost_sums) over the rollout window.
            n_episodes: number of completed episodes in the window. Logged only.
            source: label written to _lambda_update_source for debugging.
        """
        if self._lambda_update_mode != "rollout_episode_rate":
            raise RuntimeError(
                "update_lambda_from_rollout_episodes() 只用于 "
                "lambda_update_mode='rollout_episode_rate'. "
                f"当前模式: {self._lambda_update_mode!r}"
            )
        if self._replay_memory.size <= self._warmup_transitions():
            self._lambda_update_source = "rollout_episode_rate_waiting_warmup"
            return False

        selected_cost_rate = torch.as_tensor(
            float(cost_episode_rate),
            dtype=torch.float32,
            device=self._log_lambda.device,
        )
        violation = selected_cost_rate - self._cost_limit

        return self._apply_lambda_violation(
            violation,
            selected_cost_rate,
            q_c_mean=None,
            batch_cost_rate=None,
            recent_cost_rate=selected_cost_rate,
            source=f"{source}(n_ep={n_episodes})",
        )

    def _clip_gradient(self):
        # 先执行父类（mushroom actor_optimizer 配置的 clipping，若有）
        super()._clip_gradient()
        # 额外的 actor grad clip，防止 critic warmup 结束时第一次 actor 更新梯度爆炸
        if self._actor_grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self._parameters, self._actor_grad_clip)

    def _post_load(self):
        super()._post_load()
        if not hasattr(self, "_lambda_update_mode"):
            self._lambda_update_mode = "discounted_q"
        if not hasattr(self, "_lambda_qc_mean"):
            self._lambda_qc_mean = float("nan")
        if not hasattr(self, "_lambda_batch_cost_rate"):
            self._lambda_batch_cost_rate = float("nan")
        if not hasattr(self, "_lambda_recent_cost_rate"):
            self._lambda_recent_cost_rate = float("nan")
        if not hasattr(self, "_lambda_selected_cost_rate"):
            self._lambda_selected_cost_rate = float("nan")
        if not hasattr(self, "_lambda_internal_violation"):
            self._lambda_internal_violation = float("nan")
        if not hasattr(self, "_lambda_update_source"):
            self._lambda_update_source = "unknown"
        self._update_lambda_optimizer_parameters()

    def _update_lambda_optimizer_parameters(self):
        if self._lambda_optim is not None:
            if not isinstance(self._lambda_optim, optim.SGD):
                lr = self._lambda_optim.param_groups[0].get("lr", 1e-3)
                self._lambda_optim = optim.SGD([self._log_lambda], lr=lr)
            else:
                TorchUtils.update_optimizer_parameters(self._lambda_optim, [self._log_lambda])

    @property
    def _lambda(self):
        return self._log_lambda.exp()
