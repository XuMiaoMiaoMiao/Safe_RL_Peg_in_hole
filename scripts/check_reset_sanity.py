#!/usr/bin/env python3
"""One-shot reset sanity check for the dual-arm env.

This intentionally creates exactly one IsaacSim environment per process. IsaacSim
can crash when an app is started/stopped repeatedly in the same Python process.
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--default_pose_variant", choices=("easy", "harder"), default="harder")
    p.add_argument("--initial_joint_noise", type=float, default=0.0)
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main():
    args = parse_args()

    from envs import DualArmPegHoleCostEnv

    mdp = DualArmPegHoleCostEnv(
        num_envs=args.num_envs,
        headless=args.headless,
        horizon=2,
        use_axis_resid_obs=True,
        geom_stage="prepos",
        geom_d_target_neg=-0.08,
        geom_d_target_pos=0.03,
        preinsert_offset=0.08,
        geom_d_sat=0.3,
        geom_radial_sat=1.5,
        geom_d_th=0.03,
        geom_r_tip_th=0.03,
        rew_geom_d=8.0,
        rew_geom_radial_tip=8.0,
        cost_signal="clearance",
        clearance_cost_margin=0.02,
        cost_scale=1.0,
        sphere_collision_terminates=False,
        physx_collision_terminates=True,
        enable_physx_arm_collision=True,
        enable_table_collision=True,
        table_collision_terminates=True,
        table_z=0.0,
        table_clearance_hard=0.0,
        table_clearance_cost_margin=0.03,
        clearance_hard=0.0,
        proxy_arm_radius=0.065,
        proxy_ee_radius=0.04,
        keep_collision_reward_penalty=True,
        default_pose_variant=args.default_pose_variant,
        initial_joint_noise=args.initial_joint_noise,
    )
    try:
        mdp.seed(args.seed)
        mask = torch.ones(mdp._n_envs, dtype=torch.bool, device=mdp._device)
        obs, _ = mdp.reset_all(mask)
        absorbing = mdp.is_absorbing(obs)
        state = mdp.get_logging_state()

        physx_mask = getattr(mdp, "_last_physx_collision_mask", None)
        sphere_mask = getattr(mdp, "_last_sphere_collision_mask", None)
        table_mask = state.get("last_table_collision_mask")
        combined = state.get("last_collision_mask")
        min_clear = state.get("last_min_clearance")
        min_table = state.get("last_min_table_clearance")

        def mask_sum(mask_value):
            if mask_value is None:
                return "NA"
            return int(mask_value.sum().item())

        print(f"pose={args.default_pose_variant}")
        print(f"initial_joint_noise={args.initial_joint_noise}")
        print(f"num_envs={args.num_envs}")
        print(f"absorbing_sum={int(absorbing.sum().item())}")
        print(f"physx_collision_sum={mask_sum(physx_mask)}")
        print(f"sphere_collision_sum={mask_sum(sphere_mask)}")
        print(f"table_collision_sum={mask_sum(table_mask)}")
        print(f"combined_collision_sum={mask_sum(combined)}")
        if min_clear is not None:
            print(f"min_arm_clearance_m={float(min_clear.min().item()):.6f}")
            print(f"mean_arm_clearance_m={float(min_clear.mean().item()):.6f}")
        if min_table is not None:
            print(f"min_table_clearance_m={float(min_table.min().item()):.6f}")
            print(f"mean_table_clearance_m={float(min_table.mean().item()):.6f}")
    finally:
        mdp.stop()


if __name__ == "__main__":
    main()
