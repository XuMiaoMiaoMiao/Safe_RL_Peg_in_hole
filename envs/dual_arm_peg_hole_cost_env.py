"""CMDP/Lagrangian adapter for :class:`DualArmPegHoleEnv`.

`dual_arm_peg_hole_env.py` is the single source of truth for environment
mechanics: observation construction, analytic peg/hole frames, reset/setup,
collision checks, geom-stage caches, penetration metrics, success masks, and
diagnostic info. This subclass only owns the CMDP layer:

- reward formula
- cost signal selection
- logging state needed by the Lagrangian training loop
"""

import torch

from envs.dual_arm_peg_hole_env import DualArmPegHoleEnv


class DualArmPegHoleCostEnv(DualArmPegHoleEnv):
    """Thin CMDP adapter: parent env facts + separate reward/cost surface."""

    _DEFAULT_W_POS     = 1.0
    _DEFAULT_W_AXIS    = 0.5
    _DEFAULT_W_AXIS_PROGRESS = 0.0
    _DEFAULT_W_SUCCESS = 2.0
    _DEFAULT_W_ACTION  = 0.005
    _DEFAULT_W_HOME    = 0.001

    def __init__(
        self,
        rew_pos=_DEFAULT_W_POS,
        rew_axis=_DEFAULT_W_AXIS,
        rew_axis_progress=_DEFAULT_W_AXIS_PROGRESS,
        rew_success=_DEFAULT_W_SUCCESS,
        rew_action=_DEFAULT_W_ACTION,
        rew_home=_DEFAULT_W_HOME,
        **parent_kwargs,
    ):
        """
        Args:
            rew_pos:     pos_err 惩罚系数.       默认 1.0
            rew_axis:    axis_err 惩罚系数.      默认 0.5 (axis_err∈[0,2], 折合满量程≈pos项)
            rew_axis_progress: 轴对齐渐进正奖励系数. 默认 0.0
            rew_success: 成功 per-step bonus.    默认 2.0
            rew_action:  动作 L2 正则系数.        默认 0.005
            rew_home:    Home 偏差正则系数.       默认 0.001 (均匀权重, tie-breaker 量级)
            **parent_kwargs: 透传 DualArmPegHoleEnv. 几何、obs、reset、collision、
                success mask 和 info 诊断全部由父类维护.
        """
        parent_kwargs.setdefault("rew_pos", rew_pos)
        parent_kwargs.setdefault("rew_axis", rew_axis)
        parent_kwargs.setdefault("rew_success", rew_success)
        parent_kwargs.setdefault("rew_action", rew_action)
        parent_kwargs.setdefault("rew_home", rew_home)

        self._s2_w_pos     = float(rew_pos)
        self._s2_w_axis    = float(rew_axis)
        # Kept for CLI/logging compatibility with old Stage-2 Lagrangian runs.
        # The current CMDP reward borrows the parent SAC reward and does not use
        # this extra axis-progress term.
        self._s2_w_axis_progress = float(rew_axis_progress)
        self._s2_w_success = float(rew_success)
        self._s2_w_action  = float(rew_action)
        self._s2_w_home    = float(rew_home)
        # CMDP 训练默认不把 hold-N success 做成 terminal cliff. 碰撞 / geom
        # success / penetration 等环境语义仍完全走父类。
        parent_kwargs.pop("terminal_hold_bonus", None)
        super().__init__(terminal_hold_bonus=0.0, **parent_kwargs)

    # ------------------------------------------------------------------
    # Diagnostics consumed by scripts/train_sac_lagrangian.py.
    # ------------------------------------------------------------------
    def get_logging_state(self):
        """Return CMDP diagnostics consumed by training/logging code."""
        return {
            "absorb_count": self._absorb_count,
            "absorb_count_physx": self._absorb_count_physx,
            "absorb_count_sphere": self._absorb_count_sphere,
            "last_min_clearance": self._last_min_clearance,
            "last_collision_mask": self._last_collision_mask,
            "last_success_mask": self._last_success_mask,
        }

    # ------------------------------------------------------------------
    # CMDP reward: reuse the SAC task reward but leave hard safety penalties to
    # cost()/lambda. In geom mode, penetration is also treated as a constraint
    # signal, so its reward component is dropped to avoid double pressure.
    # ------------------------------------------------------------------
    def reward(self, obs, action, next_obs, absorbing):
        if self._geom_stage is not None:
            components = self._compute_geom_reward_components(next_obs)
            components["r_geom_penetration"] = torch.zeros_like(
                components["r_geom_penetration"]
            )
            r = sum(components.values())
        else:
            r = self._compute_normal_reward(next_obs)
        return self._reward_scale * r

    # ------------------------------------------------------------------
    # CMDP cost signal.
    # ------------------------------------------------------------------
    def cost(self):
        if self._cost_signal == "penetration" and self._cached_penetration_max is not None:
            return self._cached_penetration_max.to(torch.float32)
        if self._last_collision_mask is None:
            return torch.zeros(self._n_envs, dtype=torch.float32, device=self._device)
        return self._last_collision_mask.to(torch.float32)

    def _create_info_dictionary(self, obs):
        info = super()._create_info_dictionary(obs)
        info["cost"] = self.cost()
        return info
