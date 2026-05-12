"""train_sac.py / eval_sac.py / visualize_policy.py 共用的 eval 工具.

- deterministic_policy: SAC 评估时把 tanh-Gaussian 策略替换成 tanh(mu).
- compute_hold_metrics: 从 evaluate() 返回的 flatten 后的 dataset 算 hold-N
  success 与每步 in-threshold / pos_err / axis_err 统计.
  in_thresh = (pos_err < pos_th) ∧ (axis_err < axis_th); axis_th=inf 退化为
  pos-only (Stage 1 行为). 同时单独报 pos_success_rate (只看 pos<pos_th, 反映
  Stage 1 已学技能保住没) 与 axis_gate_mean (axis 项实际被门控到几成).
- parse_home_weights: argparse type=, 接受 7/14 维 float 列表 (逗号或空格分隔).
- resolve_eval_episode_count: 让 eval episode 数与 num_envs 对齐.
"""

import argparse
import math
from contextlib import contextmanager

import numpy as np
import torch


def parse_home_weights(value):
    """argparse type=: 接受 7 维(单臂, 自动复制到左右臂)或 14 维 float 列表,
    逗号或空格分隔. 例如 '1,1,1,1,0.5,0.25,0.25'.
    """
    raw = value.replace(",", " ").split()
    try:
        weights = tuple(float(x) for x in raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            "--home_weights 必须是逗号或空格分隔的 float 列表"
        ) from e
    if len(weights) not in (7, 14):
        raise argparse.ArgumentTypeError(
            f"--home_weights 必须是 7 维(单臂)或 14 维, 当前 {len(weights)} 维"
        )
    bad = [i for i, w in enumerate(weights) if not math.isfinite(w) or w < 0.0]
    if bad:
        raise argparse.ArgumentTypeError(
            f"--home_weights 必须是有限非负数, 非法索引 {bad}"
        )
    return weights


@contextmanager
def deterministic_policy(agent):
    policy = agent.policy
    original_draw_action = policy.draw_action

    def draw_action(state, internal_state=None):
        with torch.no_grad():
            mu = policy._mu_approximator.predict(state)
            action = torch.tanh(mu) * policy._delta_a + policy._central_a
        return action.detach(), None

    policy.draw_action = draw_action
    try:
        yield
    finally:
        policy.draw_action = original_draw_action


def compute_hold_metrics(dataset, mdp, hold_n_steps):
    """`hold_success` = episode 内出现长度 >= N 的连续 in-threshold 段.

    in-threshold = (pos_err < pos_th) ∧ (axis_err < axis_th). axis_th=inf 时
    退化为 pos-only (Stage 1 行为).

    依赖 VectorCore.evaluate 末尾返回的 dataset 是 flatten 后的 1D, 每个 env 的
    transitions 在结果里连续, 所以 `flatnonzero(last)` + 顺序切片就能正确分段.
    """
    _, _, _, next_state, _, last = dataset.parse(to="torch")
    pos_err, axis_err, in_thresh = mdp._compute_task_errors(next_state)
    pos_th = mdp._preinsert_success_pos_threshold
    pos_in_thresh = pos_err < pos_th
    # axis-gate 跟 reward 同公式; gate_radius=inf 时全 1.
    if math.isfinite(mdp._axis_gate_radius):
        denom = max(mdp._axis_gate_radius - pos_th, 1e-6)
        axis_gate = ((mdp._axis_gate_radius - pos_err) / denom).clamp(0.0, 1.0)
    else:
        axis_gate = torch.ones_like(pos_err)

    last_np = last.cpu().numpy().astype(bool)
    in_thresh_np = in_thresh.cpu().numpy().astype(bool)

    end_indices = np.flatnonzero(last_np)
    ep_max_holds, ep_in_thresh_rates, ep_final_in_thresh = [], [], []
    start = 0
    for end in end_indices:
        ep = in_thresh_np[start:end + 1]
        max_run, cur = 0, 0
        for flag in ep:
            cur = cur + 1 if flag else 0
            if cur > max_run:
                max_run = cur
        ep_max_holds.append(max_run)
        ep_in_thresh_rates.append(float(ep.mean()) if len(ep) else 0.0)
        ep_final_in_thresh.append(bool(ep[-1]) if len(ep) else False)
        start = end + 1

    hold_flags = np.asarray([mh >= hold_n_steps for mh in ep_max_holds], dtype=bool)
    # 条件指标: 只在 pos_in_thresh 的 timesteps 上算 axis 相关统计.
    # 如果 axis_err_in_pos_thresh_mean 长期 1.0+ 而 pos_success_rate>0, 就坐实
    # "进 pos_th 但 axis 学不会" — 这是 obs 信号不足 (axis_dot 标量缺方向信息)
    # 而不是 reward 量级问题, reward 调参治不了, 必须加 axis 向量 obs.
    pos_in_thresh_count = int(pos_in_thresh.sum().item())
    if pos_in_thresh_count > 0:
        axis_err_in_pos_thresh_mean = float(axis_err[pos_in_thresh].mean())
        axis_err_in_pos_thresh_min = float(axis_err[pos_in_thresh].min())
        axis_gate_in_pos_thresh_mean = float(axis_gate[pos_in_thresh].mean())
    else:
        axis_err_in_pos_thresh_mean = float("nan")
        axis_err_in_pos_thresh_min = float("nan")
        axis_gate_in_pos_thresh_mean = float("nan")

    return {
        "hold_success_rate": float(hold_flags.mean()) if len(hold_flags) else 0.0,
        "max_hold_mean": float(np.mean(ep_max_holds)) if ep_max_holds else 0.0,
        "in_thresh_rate": float(np.mean(ep_in_thresh_rates)) if ep_in_thresh_rates else 0.0,
        "final_in_thresh_rate": float(np.mean(ep_final_in_thresh)) if ep_final_in_thresh else 0.0,
        "pos_err_mean": float(pos_err.mean()),
        "axis_err_mean": float(axis_err.mean()),
        # Stage 1 pos 技能保住程度 — 不被 full success (pos∧axis) 掩盖.
        "pos_success_rate": float(pos_in_thresh.float().mean()),
        # axis 项实际被门控的程度. ≈0 = 还远, axis 不施压; ≈1 = 进 pos 阈, axis 满压.
        "axis_gate_mean": float(axis_gate.mean()),
        # 真正进 reward 的 axis 惩罚量级 (gate * axis_err), 反映 axis 信号强度.
        "gated_axis_penalty_mean": float((axis_gate * axis_err).mean()),
        # 条件指标 — 只看 pos_in_thresh 帧:
        "pos_in_thresh_count": pos_in_thresh_count,
        "axis_err_in_pos_thresh_mean": axis_err_in_pos_thresh_mean,
        "axis_err_in_pos_thresh_min": axis_err_in_pos_thresh_min,
        "axis_gate_in_pos_thresh_mean": axis_gate_in_pos_thresh_mean,
    }




def compute_geom_metrics(dataset, mdp, hold_n_steps):
    """Geometric preinsert metrics for geom_stage={prepos,preaxis,insert}.

    The env writes these fields from the same cached geometry used by the
    reward. The active success mask changes with mdp._geom_stage:
    prepos uses d target + radial_tip; preaxis adds radial_max + axis; insert
    uses insertion depth + radial_max + axis.
    """
    _, _, _, _, _, last = dataset.parse(to="torch")
    info = dataset.info.data
    required = (
        "geom_d", "geom_d_target", "geom_radial_tip", "geom_radial_max",
        "geom_axis_err", "geom_success_mask", "geom_prepos_mask",
        "geom_preaxis_mask", "geom_insert_mask",
    )
    missing = [k for k in required if info.get(k) is None]
    if missing:
        raise KeyError(
            f"compute_geom_metrics 需要 dataset.info.data 含 {required}, 缺 {missing}. "
            "确认 mdp._geom_stage 已启用且 env 写入了 geom info."
        )

    def _to_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    d = _to_tensor(info["geom_d"])
    d_target = _to_tensor(info["geom_d_target"])
    radial_tip = _to_tensor(info["geom_radial_tip"])
    radial_max = _to_tensor(info["geom_radial_max"])
    axis_err = _to_tensor(info["geom_axis_err"])
    success = _to_tensor(info["geom_success_mask"]).to(torch.bool)
    prepos = _to_tensor(info["geom_prepos_mask"]).to(torch.bool)
    preaxis = _to_tensor(info["geom_preaxis_mask"]).to(torch.bool)
    insert = _to_tensor(info["geom_insert_mask"]).to(torch.bool)
    # penetration_max: codex 2026-05-11 trajectory-level 几何穿模量, > 0 = 物理穿模.
    # 老 ckpt 没写过这个 field; 兼容缺失情况, fallback 全 0 (= 不报告穿模 stats).
    penetration_raw = info.get("geom_penetration_max")
    if penetration_raw is None:
        penetration = torch.zeros_like(d)
    else:
        penetration = _to_tensor(penetration_raw)

    last_np = last.cpu().numpy().astype(bool)
    success_np = success.cpu().numpy().astype(bool)
    insert_np = insert.cpu().numpy().astype(bool)
    end_indices = np.flatnonzero(last_np)

    # entry/final per-episode 诊断 (codex 2026-05-11): 单看 *_min 信息量不够,
    # 区分不出 "全程斜" 跟 "斜插再调直". insert_entry = 每 episode 第一次 active
    # success mask=True 的那一步 state (回答 "进入时是否对齐"); final = 每 ep
    # 最后一步 state (回答 "终态是否真的稳定").
    # 注意: 用 `success_np` (active mask, 已按 geom_stage 选 prepos/preaxis/insert)
    # 而非永远用 insert_np, 这样 prepos/preaxis 阶段也能正确报 entry 数据.
    d_np = d.cpu().numpy()
    d_target_np = d_target.cpu().numpy()
    radial_max_np = radial_max.cpu().numpy()
    axis_err_np = axis_err.cpu().numpy()
    penetration_np = penetration.cpu().numpy()

    ep_max_runs = []
    ep_success_rates = []
    ep_final_success = []
    entry_d = []
    entry_d_err = []
    entry_radial_max = []
    entry_axis_err = []
    entry_penetration = []
    final_d = []
    final_d_err = []
    final_radial_max = []
    final_axis_err = []
    final_penetration = []
    n_ep_with_entry = 0
    start = 0
    for end in end_indices:
        sl = slice(start, end + 1)
        ep = success_np[sl]
        max_run, cur = 0, 0
        for flag in ep:
            cur = cur + 1 if flag else 0
            if cur > max_run:
                max_run = cur
        ep_max_runs.append(max_run)
        ep_success_rates.append(float(ep.mean()) if len(ep) else 0.0)
        ep_final_success.append(bool(ep[-1]) if len(ep) else False)
        # final state: 该 episode 最后一步
        if end >= start:
            final_d.append(float(d_np[end]))
            final_d_err.append(abs(float(d_np[end]) - float(d_target_np[end])))
            final_radial_max.append(float(radial_max_np[end]))
            final_axis_err.append(float(axis_err_np[end]))
            final_penetration.append(float(penetration_np[end]))
        # entry state: 该 episode 内首次满足 active success mask 的那一步
        ep_idx = np.flatnonzero(ep)
        if len(ep_idx) > 0:
            entry_global = start + int(ep_idx[0])
            entry_d.append(float(d_np[entry_global]))
            entry_d_err.append(abs(float(d_np[entry_global]) - float(d_target_np[entry_global])))
            entry_radial_max.append(float(radial_max_np[entry_global]))
            entry_axis_err.append(float(axis_err_np[entry_global]))
            entry_penetration.append(float(penetration_np[entry_global]))
            n_ep_with_entry += 1
        start = end + 1

    hold_flags = np.asarray([r >= hold_n_steps for r in ep_max_runs], dtype=bool)
    d_err = torch.abs(d - d_target)

    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    def _max(xs):
        return float(np.max(xs)) if xs else 0.0

    return {
        "geom_step_rate": float(success.float().mean()),
        "geom_hold_rate": float(hold_flags.mean()) if len(hold_flags) else 0.0,
        "geom_max_run_mean": float(np.mean(ep_max_runs)) if ep_max_runs else 0.0,
        "geom_ep_success_rate_mean": (
            float(np.mean(ep_success_rates)) if ep_success_rates else 0.0
        ),
        "geom_final_success_rate": (
            float(np.mean(ep_final_success)) if ep_final_success else 0.0
        ),
        "geom_prepos_step_rate": float(prepos.float().mean()),
        "geom_preaxis_step_rate": float(preaxis.float().mean()),
        "geom_insert_step_rate": float(insert.float().mean()),
        "geom_d_target_mean": float(d_target.float().mean()),
        "geom_d_err_mean": float(d_err.float().mean()),
        "geom_d_err_min": float(d_err.float().min()),
        "geom_radial_tip_mean": float(radial_tip.float().mean()),
        "geom_radial_tip_min": float(radial_tip.float().min()),
        "geom_radial_max_mean": float(radial_max.float().mean()),
        "geom_radial_max_min": float(radial_max.float().min()),
        "geom_axis_err_mean": float(axis_err.float().mean()),
        "geom_axis_err_min": float(axis_err.float().min()),
        # penetration stats (codex 2026-05-11). pen_max_mean / max 反映整个数据集
        # 的穿模量分布. clean_rate = step 级别 penetration < 1e-4 的比例
        # (1e-4 是数值噪声容忍; 真穿模通常 > 1e-3). pen_in_insert_* 仅在
        # 当前 active mask=True 的 step 上聚合 — 用来回答"被算成功的 step 里
        # 有多少其实在穿模".
        "geom_pen_max_mean": float(penetration.float().mean()),
        "geom_pen_max_max":  float(penetration.float().max()),
        "geom_clean_step_rate": float((penetration < 1e-4).float().mean()),
        "geom_pen_in_active_mean": (
            float(penetration[success].float().mean())
            if success.any() else 0.0
        ),
        "geom_pen_in_active_max": (
            float(penetration[success].float().max())
            if success.any() else 0.0
        ),
        # entry / final 诊断 (codex 2026-05-11). entry = 每 ep 首次 active success
        # 那一步; final = 每 ep 最后一步. 用来分辨 "全程都好" / "tilt-then-align" /
        # "approach-then-dwell" 三种 mode.
        "geom_n_ep_with_entry": int(n_ep_with_entry),
        "geom_entry_d_mean":          _mean(entry_d),
        "geom_entry_d_err_mean":      _mean(entry_d_err),
        "geom_entry_radial_max_mean": _mean(entry_radial_max),
        "geom_entry_radial_max_max":  _max(entry_radial_max),
        "geom_entry_axis_err_mean":   _mean(entry_axis_err),
        "geom_entry_axis_err_max":    _max(entry_axis_err),
        "geom_entry_penetration_mean": _mean(entry_penetration),
        "geom_entry_penetration_max":  _max(entry_penetration),
        "geom_final_d_mean":          _mean(final_d),
        "geom_final_d_err_mean":      _mean(final_d_err),
        "geom_final_radial_max_mean": _mean(final_radial_max),
        "geom_final_radial_max_max":  _max(final_radial_max),
        "geom_final_axis_err_mean":   _mean(final_axis_err),
        "geom_final_axis_err_max":    _max(final_axis_err),
        "geom_final_penetration_mean": _mean(final_penetration),
        "geom_final_penetration_max":  _max(final_penetration),
    }


def compute_cost_metrics(dataset, n_eval_episodes):
    """从 eval flatten dataset 的 info.data["cost"] 算 cost_rate / per-ep cost sum.

    依赖 env._create_info_dictionary 把 cost 写进 step_info; flatten 后顺序与
    reward 对齐. cost = 0/1 per-step collision indicator.
    """
    import torch
    cost = dataset.info.data.get("cost")
    if cost is None:
        return {"cost_rate": float("nan"), "cost_episode_sum_mean": float("nan")}
    cost_t = cost if isinstance(cost, torch.Tensor) else torch.as_tensor(cost)
    cost_rate = float(cost_t.float().mean())
    cost_episode_sum_mean = float(cost_t.float().sum()) / max(n_eval_episodes, 1)
    return {"cost_rate": cost_rate, "cost_episode_sum_mean": cost_episode_sum_mean}


def resolve_eval_episode_count(requested_episodes, num_envs, arg_name):
    """评估 episode 数与 vectorized env 对齐.

    `VectorCore.evaluate(n_episodes=...)` 在尾批不足时会把 inactive env
    teleport away. 为了避免渲染里出现"飞天"机器人, 统一要求 eval episode 数
    是 num_envs 的整数倍.
    """
    if requested_episodes is None:
        return num_envs
    if requested_episodes < num_envs:
        raise ValueError(
            f"{arg_name} ({requested_episodes}) 不能小于 num_envs ({num_envs}). "
            "否则 evaluate 的 inactive env 会被 teleport away."
        )
    if requested_episodes % num_envs != 0:
        raise ValueError(
            f"{arg_name} ({requested_episodes}) 必须能被 num_envs ({num_envs}) 整除, "
            "否则最后一批会留下 inactive env 被 teleport away."
        )
    return requested_episodes


def warmstart_actor_with_partial_copy(new_agent, ckpt_path):
    """Stage 1→2 (整块) / Stage 2→3 (34→41 partial) 的 actor warm-start.

    new_agent 已经 cold-create 完毕, 维度匹配当前 env (geom_stage = 41D).
    ckpt_path 指向 actor 来源 (通常 Stage 2 ckpt, 34D).

    行为:
        - obs 维度一致 (S1→2, S2→2): 整块 set_weights actor (mu+sigma).
        - new_h1_in > old_h1_in (S2→S3): 第一层前 old_h1_in 列拷, 新增列
          zero-init (epoch 0 actor 输出严格等价 old). h2/out/全部 bias 整块拷.
        - new_h1_in < old_h1_in: raise (降维无合理 partial copy 语义).

    critic / alpha / replay 不动 — 留给调用方决定 (训练用全冷, dry-run 不需要).
    """
    # mushroom 必须在 IsaacSim 启动后导入, lazy import 在函数体内.
    import torch
    from mushroom_rl.core import Agent

    old_agent = Agent.load(str(ckpt_path))
    old_h1_in = old_agent.policy._mu_approximator.model.network._h1.weight.shape[1]
    new_h1_in = new_agent.policy._mu_approximator.model.network._h1.weight.shape[1]
    if old_h1_in == new_h1_in:
        new_agent.policy._mu_approximator.set_weights(
            old_agent.policy._mu_approximator.get_weights()
        )
        new_agent.policy._sigma_approximator.set_weights(
            old_agent.policy._sigma_approximator.get_weights()
        )
        mode = f"obs 维度一致 ({old_h1_in}D), 整块继承 actor"
    elif new_h1_in > old_h1_in:
        for new_app, old_app in [
            (new_agent.policy._mu_approximator,
             old_agent.policy._mu_approximator),
            (new_agent.policy._sigma_approximator,
             old_agent.policy._sigma_approximator),
        ]:
            new_net = new_app.model.network
            old_net = old_app.model.network
            with torch.no_grad():
                new_net._h1.weight[:, :old_h1_in].copy_(old_net._h1.weight)
                new_net._h1.weight[:, old_h1_in:].zero_()
                new_net._h1.bias.copy_(old_net._h1.bias)
                new_net._h2.weight.copy_(old_net._h2.weight)
                new_net._h2.bias.copy_(old_net._h2.bias)
                new_net._out.weight.copy_(old_net._out.weight)
                new_net._out.bias.copy_(old_net._out.bias)
        mode = (f"obs {old_h1_in}→{new_h1_in}D partial, "
                f"前 {old_h1_in} 列继承, 新 {new_h1_in - old_h1_in} 列 zero-init")
    else:
        raise ValueError(
            f"warm-start obs 维度倒退: 新 {new_h1_in}D < 旧 {old_h1_in}D, "
            "降维没有合理的 partial copy 语义."
        )
    del old_agent
    return mode
