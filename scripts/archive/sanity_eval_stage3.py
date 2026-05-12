"""Stage 3 sanity eval: 钉死 d sign convention + 验证 collision 行为 (无训练).

加载 Stage 2 oneshot ckpt, 跑 N episodes, monkey-patch is_absorbing, 在每步从
_compute_preinsert_errors 读 cache 量, 同时计算 radial_max (peg 沿轴 4 个采样点
的 max radial), 最后 print 几何量分布 + collision 计数.

预期 (基于 envs/dual_arm_peg_hole_env.py:516 + :811:
    preinsert_target = hole_entry + preinsert_offset * hole_axis,
    DEFAULT_PREINSERT_OFFSET = 0.05):
    preinsert (pos_err < pos_th): axial_dist ≈ +preinsert_offset (≈ +0.05)
    entry plane:                   axial_dist ≈ 0
    inserted:                      axial_dist < 0
所以 Stage 3 reward 用 d = -axial_dist (d>0 = inserted).

如果实测 axial_dist 在 preinsert 区域 中位数 ≠ +preinsert_offset (符号反 / 量级
不对), Stage 3 reward 不能直接写 — 先查 hole_axis_local / preinsert_offset / quat
是否飘了.

运行示例 (Stage 2 oneshot 训练参数):
    python scripts/archive/sanity_eval_stage3.py \
        --agent_path results/S2_oneshot_ep194_seed0_best_agent.msh \
        --use_axis_resid_obs \
        --rew_axis 1.0 --rew_pos_success 1.0 --rew_success 4.0 \
        --rew_home 0.0005 --home_weights 1,1,1,1,0.75,0.5,0.5 \
        --success_axis_threshold 0.40 --axis_gate_radius 0.40 \
        --headless
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Archived under scripts/archive/, so parents[2] is the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts._eval_utils import (
    deterministic_policy,
    parse_home_weights,
    resolve_eval_episode_count,
)


# 沿 peg 反向 (从 tip 往 base) 的采样偏移. peg 总长 7cm (_PEG_HEIGHT=0.070),
# 不要超过 -0.06, 否则采到 EE coupler 段虚拟杆 (见 feedback_bimanual_stage3_traps.md).
PEG_RADIAL_SAMPLE_OFFSETS = (0.0, -0.02, -0.04, -0.06)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--agent_path", type=str,
                   default=str(PROJECT_ROOT
                               / "results/S2_oneshot_ep194_seed0_best_agent.msh"))
    p.add_argument("--n_episodes", type=int, default=None,
                   help="默认取 num_envs (16). 必须能被 num_envs 整除.")
    p.add_argument("--num_envs", type=int, default=16,
                   help="必须 >=2 (num_envs=1 触发 cloner bug).")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--stochastic", action="store_true",
                   help="用 SAC 采样而不是 deterministic tanh(mu).")

    # Stage 2 oneshot 训练参数 — 必须与 train 一致, 否则 hold/in_thresh 数字无意义.
    # axial_dist / radial 是 fresh 几何量, 不依赖这些; 但保留对齐方便看 J/in_thresh.
    p.add_argument("--use_axis_resid_obs", action="store_true",
                   help="34 维 obs (Stage 2 oneshot 必传, 否则 actor 维度对不上).")
    p.add_argument("--rew_axis", type=float, default=None)
    p.add_argument("--rew_pos_success", type=float, default=None)
    p.add_argument("--rew_success", type=float, default=None)
    p.add_argument("--rew_home", type=float, default=None)
    p.add_argument("--home_weights", type=parse_home_weights, default=None)
    p.add_argument("--success_axis_threshold", type=float, default=None)
    p.add_argument("--axis_gate_radius", type=float, default=None)
    p.add_argument("--preinsert_success_pos_threshold", type=float, default=None)
    p.add_argument("--preinsert_offset", type=float, default=None)
    p.add_argument("--initial_joint_noise", type=float, default=None)
    p.add_argument("--clearance_hard", type=float, default=None,
                   help="sphere-proxy 自碰撞兜底. 关闭传 -inf.")
    p.add_argument("--proxy_arm_radius", type=float, default=None)
    p.add_argument("--proxy_ee_radius", type=float, default=None)

    # Stage 3 默认开 EE exclude (peg-hole 接触不被 hard absorb 误杀).
    p.add_argument("--no_exclude_ee", action="store_true",
                   help="对照实验: 关掉 EE exclude. 默认开 (Stage 3 行为).")
    return p.parse_args()


def _summarize(arr, name, unit=""):
    if arr.size == 0:
        return f"  {name:18s} <empty>"
    qs = np.percentile(arr, [0, 5, 25, 50, 75, 95, 100])
    return (f"  {name:18s} n={arr.size:6d}  "
            f"min={qs[0]:+.4f}{unit}  p5={qs[1]:+.4f}  p25={qs[2]:+.4f}  "
            f"med={qs[3]:+.4f}  p75={qs[4]:+.4f}  p95={qs[5]:+.4f}  "
            f"max={qs[6]:+.4f}  mean={arr.mean():+.4f}")


def main():
    args = parse_args()
    args.n_episodes = resolve_eval_episode_count(
        args.n_episodes, args.num_envs, "--n_episodes")

    from envs import DualArmPegHoleEnv

    env_kwargs = dict(num_envs=args.num_envs, headless=args.headless)
    for key in ("rew_axis", "rew_pos_success", "rew_success", "rew_home",
                "home_weights", "success_axis_threshold", "axis_gate_radius",
                "preinsert_success_pos_threshold", "preinsert_offset",
                "initial_joint_noise", "clearance_hard",
                "proxy_arm_radius", "proxy_ee_radius"):
        v = getattr(args, key)
        if v is not None:
            env_kwargs[key] = v
    if args.use_axis_resid_obs:
        env_kwargs["use_axis_resid_obs"] = True
    if not args.no_exclude_ee:
        env_kwargs["exclude_ee_from_physx_self_collision"] = True
    print(f"[SANITY ENV kwargs] {env_kwargs}")

    mdp = DualArmPegHoleEnv(**env_kwargs)
    print(f"[SANITY] preinsert_offset = {mdp._preinsert_offset:.4f}m  "
          f"=> 预期 in_preinsert 区域 axial_dist ≈ +{mdp._preinsert_offset:.4f}")
    print(f"[SANITY] clearance_hard   = {mdp._clearance_hard}  "
          f"(finite => sphere-proxy 兜底启用)")
    print(f"[SANITY] exclude_ee       = {not args.no_exclude_ee}")

    from mushroom_rl.core import Agent, VectorCore
    agent = Agent.load(args.agent_path)
    print(f"[SANITY] loaded agent from {args.agent_path}")

    buf = {
        "axial_dist": [], "radial_err": [], "radial_max": [],
        "axis_err": [], "pos_err": [],
        "collision": [], "physx_collision": [], "sphere_collision": [],
        "min_clearance": [],
    }
    clearance_hard_finite = math.isfinite(mdp._clearance_hard)

    original_is_absorbing = mdp.is_absorbing

    def hooked_is_absorbing(obs):
        # 让 env 跑标准逻辑 (collision / cached_errors / counters), 之后 _last_*
        # 都已写入当前 step 值.
        result = original_is_absorbing(obs)

        # fresh 几何量 (visualize_policy 同款做法, 与 cached pos_vec/axis_err 同源).
        frames = mdp.get_preinsert_frames()
        errs = mdp._compute_preinsert_errors(frames)
        peg_tip = frames["peg_tip_pos"]
        peg_axis = frames["peg_axis"]
        hole_entry = frames["hole_entry_pos"]
        hole_axis = frames["hole_axis"]

        # radial_max: 沿 peg 反向 4 个采样点, 各自到 hole 中轴线的径向距离, 取 max.
        # peg_axis / hole_axis 已 unit (env 在 _create_observation/get_preinsert_frames
        # 里 normalize 过). offset<0 = 从 tip 往 peg base 走.
        rad_max = torch.zeros_like(errs["radial_err"])
        for off in PEG_RADIAL_SAMPLE_OFFSETS:
            sample_pos = peg_tip + off * peg_axis
            d_s = sample_pos - hole_entry
            axial_s = torch.sum(d_s * hole_axis, dim=-1, keepdim=True)
            radial_v_s = d_s - axial_s * hole_axis
            rad_s = torch.norm(radial_v_s, dim=-1)
            rad_max = torch.maximum(rad_max, rad_s)

        # collision 拆分: env 内部已合并, 这里靠 _last_min_clearance 反推 sphere 部分.
        coll = mdp._last_collision_mask
        if clearance_hard_finite:
            sphere = mdp._last_min_clearance < mdp._clearance_hard
        else:
            sphere = torch.zeros_like(coll)
        physx = coll & ~sphere

        buf["axial_dist"].append(errs["axial_dist"].cpu().numpy())
        buf["radial_err"].append(errs["radial_err"].cpu().numpy())
        buf["radial_max"].append(rad_max.cpu().numpy())
        buf["axis_err"].append(errs["axis_err"].cpu().numpy())
        buf["pos_err"].append(mdp._last_pos_err.cpu().numpy())
        buf["collision"].append(coll.cpu().numpy().astype(np.bool_))
        buf["physx_collision"].append(physx.cpu().numpy().astype(np.bool_))
        buf["sphere_collision"].append(sphere.cpu().numpy().astype(np.bool_))
        buf["min_clearance"].append(mdp._last_min_clearance.cpu().numpy())

        return result

    mdp.is_absorbing = hooked_is_absorbing

    core = VectorCore(agent, mdp)
    if args.stochastic:
        core.evaluate(n_episodes=args.n_episodes,
                      render=not args.headless, quiet=False)
    else:
        with deterministic_policy(agent):
            core.evaluate(n_episodes=args.n_episodes,
                          render=not args.headless, quiet=False)

    for k in buf:
        buf[k] = np.concatenate(buf[k]) if buf[k] else np.array([])

    pos_th = mdp._preinsert_success_pos_threshold
    pre_mask = buf["pos_err"] < pos_th

    print("\n" + "=" * 80)
    print(f"[STATS] total steps = {buf['axial_dist'].size}  "
          f"in_preinsert (pos_err<{pos_th:.3f}m): {int(pre_mask.sum())} "
          f"({100 * pre_mask.mean():.1f}%)")
    print("=" * 80)

    print("\n[axial_dist] raw 几何量 (m, peg_tip - hole_entry 投影到 hole_axis):")
    print(_summarize(buf["axial_dist"], "all_steps", "m"))
    if pre_mask.any():
        print(_summarize(buf["axial_dist"][pre_mask], "in_preinsert", "m"))

    expected = mdp._preinsert_offset
    print(f"\n  预期: in_preinsert 区域 axial_dist 中位数 ≈ +{expected:.4f}m")
    if pre_mask.any():
        med = float(np.median(buf["axial_dist"][pre_mask]))
        gap = med - expected
        sign_matches_expected = (med > 0)
        magnitude_ok = abs(gap) < 0.02
        print(f"  实测: median = {med:+.4f}m  (gap to expected = {gap:+.4f}m)")
        if sign_matches_expected and magnitude_ok:
            print(f"  ✓ sign + 量级都符合预期 => Stage 3 钉死 d = -axial_dist (d>0 = inserted)")
        elif sign_matches_expected and not magnitude_ok:
            print(f"  ⚠️  sign 对但量级偏 ({gap:+.4f}m) - 检查 preinsert_offset 是否被 CLI 改过")
        else:
            print(f"  ❌ sign 跟预期反 - 检查 hole_axis_local / quat 朝向, 不要直接写 reward")

    print("\n[radial_err]  tip 单点径向 (m)")
    print(_summarize(buf["radial_err"], "all_steps", "m"))
    if pre_mask.any():
        print(_summarize(buf["radial_err"][pre_mask], "in_preinsert", "m"))

    print(f"\n[radial_max]  peg 沿轴 {len(PEG_RADIAL_SAMPLE_OFFSETS)} 采样点 "
          f"{PEG_RADIAL_SAMPLE_OFFSETS} 取 max (m)")
    print(_summarize(buf["radial_max"], "all_steps", "m"))
    if pre_mask.any():
        print(_summarize(buf["radial_max"][pre_mask], "in_preinsert", "m"))

    diff = buf["radial_max"] - buf["radial_err"]
    print(f"\n[radial_max - radial_err]  杆身 vs tip 径向差 (反映轴向倾斜):")
    print(_summarize(diff, "all_steps", "m"))
    print("  => 显著 (>1cm) 说明 tip 准但杆身斜, success 用 radial_max 比 tip-only 严, 是对的.")

    print("\n[axis_err]  1 + cos(peg_axis, hole_axis), 0=完美反向对齐, 2=同向")
    print(_summarize(buf["axis_err"], "all_steps"))
    if pre_mask.any():
        print(_summarize(buf["axis_err"][pre_mask], "in_preinsert"))

    print("\n[collision] per-step 0/1 indicator")
    print(f"  total       rate = {buf['collision'].mean():.4f}  "
          f"({int(buf['collision'].sum())} / {buf['collision'].size} steps)")
    print(f"  PhysX 部分  rate = {buf['physx_collision'].mean():.4f}  "
          f"({int(buf['physx_collision'].sum())})")
    print(f"  sphere 部分 rate = {buf['sphere_collision'].mean():.4f}  "
          f"({int(buf['sphere_collision'].sum())})")
    if buf['collision'].sum() == 0:
        print("  => 0 collision: peg 没碰到 hole 也没双臂自撞. "
              "Stage 2 ckpt axis_in_pos_mean=0.41 加上 EE exclude, peg 大概率"
              "在 preinsert 锥里 hover, 不会推到 entry plane. 这是预期, "
              "不代表 Stage 3 不会撞.")
    elif buf['physx_collision'].sum() > 0 and buf['sphere_collision'].sum() == 0:
        print("  => 仅 PhysX 触发, sphere-proxy 无介入. 多半是 peg-hole 真接触 "
              "(EE exclude 已生效). 看 trajectory 确认是 hole rim 接触还是别的.")

    print(f"\n[min_clearance] sphere-proxy 双臂最近距离 (m, clearance_hard={mdp._clearance_hard})")
    print(_summarize(buf["min_clearance"], "all_steps", "m"))

    print("\n" + "=" * 80)
    print("[VERDICT]")
    if pre_mask.any():
        med = float(np.median(buf["axial_dist"][pre_mask]))
        if med > 0 and abs(med - expected) < 0.02:
            print(f"  ✓ d = -axial_dist  (preinsert 区域 axial_dist median = {med:+.4f}, "
                  f"预期 +{expected:.4f})")
            print(f"  ✓ Stage 3 reward / 1_insert / clip(d, ...) 用此约定")
        else:
            print(f"  ❌ axial_dist 在 preinsert 区域 median = {med:+.4f}, 预期 +{expected:.4f}")
            print(f"  ❌ 不要直接写 Stage 3 reward, 先排查上面 ⚠️/❌ 提示项")
    else:
        print(f"  ⚠️  无 in_preinsert 帧 (Stage 2 ckpt 没进 pos_th={pos_th}); "
              "扩大 n_episodes 或检查 ckpt")
    print("=" * 80)

    mdp.stop()


if __name__ == "__main__":
    main()
