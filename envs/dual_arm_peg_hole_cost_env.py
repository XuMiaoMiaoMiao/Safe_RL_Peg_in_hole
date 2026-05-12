"""DualArmPegHoleCostEnv — Lagrangian SAC 专用子类.

继承 DualArmPegHoleEnv 的所有几何/物理/obs/终止逻辑，仅覆写 reward() 和
_create_info_dictionary()，将碰撞信号从 reward 剥离到独立 cost 约束。

标准 SAC (DualArmPegHoleEnv):
    collision → is_absorbing=True, reward = r_min/(1-γ) ≈ -200

Lagrangian SAC (DualArmPegHoleCostEnv):
    collision → is_absorbing=True, reward = shaped reward（不加大负奖励）
                                    cost   = 1.0（per-step 约束信号）

碰撞仍然终止 episode（is_absorbing 不变），但碰撞惩罚完全由 λ·Q_C 承担，
reward critic 只学任务进展。两种环境共用相同 obs / absorbing / 网络结构，
仅 reward/cost 结构不同，构成严格对照实验。

cost 流向:
    cost() → _create_info_dictionary() → dataset.info.data["cost"]
    → ConstrainedReplayMemory.add() → SACLagrangian.fit()

当前 USD 下 PhysX 自碰撞不触发，cost 等价于纯 sphere-proxy 信号；
未来 PhysX 恢复后 cost 自然包含两种碰撞，无需改动此文件。
"""

import torch

from envs.dual_arm_peg_hole_env import DualArmPegHoleEnv


class DualArmPegHoleCostEnv(DualArmPegHoleEnv):

    def reward(self, obs, action, next_obs, absorbing):
        # 碰撞步不给 r_min/(1-γ) 惩罚——碰撞代价由 λ·Q_C 承担，reward critic 只看任务进展。
        normal = self._compute_normal_reward(next_obs)
        r = torch.where(
            self._last_hold_done_mask,
            normal + self._terminal_hold_bonus,
            normal,
        )
        return self._reward_scale * r

    def cost(self):
        # episode 第一步之前 _last_collision_mask 尚未写入，返回全零避免 KeyError。
        if self._last_collision_mask is None:
            return torch.zeros(self._n_envs, dtype=torch.float32, device=self._device)
        return self._last_collision_mask.float()

    def _create_info_dictionary(self, obs):
        # 框架 hook：将 cost 注入 dataset.info，供 ConstrainedReplayMemory 读取。
        return {"cost": self.cost()}
