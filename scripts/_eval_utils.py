"""train_sac.py 与 eval_sac.py 共用的 eval 工具.

- deterministic_policy: SAC 评估时把 tanh-Gaussian 策略替换成 tanh(mu).
- compute_hold_metrics: 从 evaluate() 返回的 flatten 后的 dataset 算 hold-N
  success 与每步 in-threshold / pos_err / axis_err 统计.
  in_thresh = (pos_err < pos_th) ∧ (axis_err < axis_th); axis_th=inf 退化为
  pos-only (M1' 行为). M2c 时通过 dataset.get_info("min_clearance") 读
  env._create_info_dictionary 注入的 clearance, 走 mushroom 同一套 mask.
  M2d 额外输出 final-window 稳停指标: 最后 N 步全满足 gate 才算成功,
  几何均值按最后窗口全体步统计.
- summarize_clearance: 从 dataset.get_info("min_clearance") 算分布 metrics
  (mean / p10 / min / 4 阈值 per-step rate). 与 dataset 同步对齐, 不会包含
  reset obs.
- summarize_approach: 从 dataset.info 的 axial_dist / axial_off / radial_err
  算 M2d approach 几何分布.
"""

from contextlib import contextmanager

import numpy as np
import torch


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


def _max_consecutive_run(flag_array):
    max_run, cur = 0, 0
    for flag in flag_array:
        cur = cur + 1 if flag else 0
        if cur > max_run:
            max_run = cur
    return max_run


def _extract_info_array(dataset, name):
    """从 dataset.get_info(name) 拉 flatten 后的 1D numpy 数组.

    env._create_info_dictionary 每个 transition 后注入 [n_envs] tensor,
    VectorizedDataset.flatten() 用 mask 把它和 next_state / last 同步对齐.
    返回 None 如果该字段不在 info 里.
    """
    try:
        value = dataset.get_info(name)
    except (KeyError, AttributeError):
        return None
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _extract_clearance_from_info(dataset):
    """从 dataset.get_info("min_clearance") 拉 flatten 后的 1D numpy 数组."""
    return _extract_info_array(dataset, "min_clearance")


def compute_hold_metrics(dataset, mdp, hold_n_steps,
                         clearance_threshold=None,
                         axial_threshold=None,
                         radial_threshold=None,
                         final_window_steps=30):
    """`hold_success` = episode 内出现长度 >= N 的连续 in-threshold 段.

    pos∧axis 路径 (一直保留, 老 metric):
        in_thresh_pa = (pos_err < pos_th) ∧ (axis_err < axis_th)
        hold_success_rate = per-episode max_run(in_thresh_pa) >= N 的比例

    M2c clearance-aware 路径 (clearance_threshold 给且 dataset.info 含
    "min_clearance" 时启用):
        in_thresh_full = in_thresh_pa ∧ (clearance >= clearance_threshold)
        hold_success_rate_with_clearance = per-episode max_run(in_thresh_full) >= N

    M2d approach 路径 (axial/radial threshold 至少一个给时启用):
        in_thresh_m2d = axis_ok ∧ axial_ok ∧ radial_ok ∧ clearance_ok
        注意: M2d 有意不使用 pos_err 球形 gate; axial_off/radial_err 来自
        dataset.info, 与 clearance 一样走 mushroom flatten mask.

    M2d final-window 路径:
        final_window_success_rate_m2d 要求每个 episode 最后 final_window_steps
        步全部满足 in_thresh_m2d. final_window_* 几何均值基于最后窗口全体步,
        不筛选 in-threshold 步, 避免把"偶尔进阈"的扫过解统计得过于乐观.
    """
    final_window_steps = int(final_window_steps)
    if final_window_steps < 1:
        raise ValueError(f"final_window_steps 必须 >= 1, 传入 {final_window_steps}")

    _, _, _, next_state, _, last = dataset.parse(to="torch")
    pos_err, axis_err, in_thresh_pa = mdp._compute_task_errors(next_state)
    approach_from_obs = None
    if hasattr(mdp, "_compute_approach_errors"):
        approach_from_obs = mdp._compute_approach_errors(next_state)

    last_np = last.cpu().numpy().astype(bool)
    in_thresh_pa_np = in_thresh_pa.cpu().numpy().astype(bool)
    axis_err_np = axis_err.cpu().numpy().reshape(-1)
    axis_ok_np = (
        axis_err.cpu().numpy() < mdp._success_axis_threshold
    ).astype(bool)

    in_thresh_full_np = None
    in_thresh_m2d_np = None
    clearance_pass_rate = float("nan")
    clearance_ok_np = None
    clr_flat = None
    if clearance_threshold is not None:
        clr_flat = _extract_clearance_from_info(dataset)
        if clr_flat is None:
            raise RuntimeError(
                "compute_hold_metrics: clearance_threshold 已传, 但 dataset.info 里"
                "没有 'min_clearance' 字段. M2c 任一 (rew_clearance>0 / "
                "clearance_soft 有限 / clearance_hard 有限) 均未启用 — 检查 CLI, "
                "或 metric-only 用 --log_clearance."
            )
        if clr_flat.size != in_thresh_pa_np.size:
            raise RuntimeError(
                f"compute_hold_metrics: clearance flat len {clr_flat.size} != "
                f"in_thresh len {in_thresh_pa_np.size}. dataset.info mask 对齐异常."
            )
        clearance_ok_np = (clr_flat >= clearance_threshold)
        in_thresh_full_np = in_thresh_pa_np & clearance_ok_np
        clearance_pass_rate = float(clearance_ok_np.mean())

    axial_flat = radial_flat = None
    axial_pass_rate = radial_pass_rate = float("nan")
    if axial_threshold is not None or radial_threshold is not None:
        in_thresh_m2d_np = axis_ok_np.copy()
        if axial_threshold is not None:
            if approach_from_obs is not None:
                axial_flat = (
                    approach_from_obs["axial_off"].detach().cpu().numpy().reshape(-1)
                )
            else:
                axial_flat = _extract_info_array(dataset, "axial_off")
                if axial_flat is None:
                    raise RuntimeError(
                        "compute_hold_metrics: axial_threshold 已传, 但 next_state "
                        "没有 M2d obs 且 dataset.info 里没有 'axial_off'. "
                        "检查 --use_m2d_obs / env._m2d_active / --log_axial_radial."
                    )
            if axial_flat.size != in_thresh_pa_np.size:
                raise RuntimeError(
                    f"compute_hold_metrics: axial_off flat len {axial_flat.size} != "
                    f"in_thresh len {in_thresh_pa_np.size}. dataset.info mask 对齐异常."
                )
            axial_ok_np = axial_flat < axial_threshold
            axial_pass_rate = float(axial_ok_np.mean())
            in_thresh_m2d_np &= axial_ok_np
        if radial_threshold is not None:
            if approach_from_obs is not None:
                radial_flat = (
                    approach_from_obs["radial_err"].detach().cpu().numpy().reshape(-1)
                )
            else:
                radial_flat = _extract_info_array(dataset, "radial_err")
                if radial_flat is None:
                    raise RuntimeError(
                        "compute_hold_metrics: radial_threshold 已传, 但 next_state "
                        "没有 M2d obs 且 dataset.info 里没有 'radial_err'. "
                        "检查 --use_m2d_obs / env._m2d_active / --log_axial_radial."
                    )
            if radial_flat.size != in_thresh_pa_np.size:
                raise RuntimeError(
                    f"compute_hold_metrics: radial_err flat len {radial_flat.size} != "
                    f"in_thresh len {in_thresh_pa_np.size}. dataset.info mask 对齐异常."
                )
            radial_ok_np = radial_flat < radial_threshold
            radial_pass_rate = float(radial_ok_np.mean())
            in_thresh_m2d_np &= radial_ok_np
        if clearance_ok_np is not None:
            in_thresh_m2d_np &= clearance_ok_np

    end_indices = np.flatnonzero(last_np)
    ep_max_holds_pa = []
    ep_max_holds_full = []
    ep_max_holds_m2d = []
    ep_in_thresh_rates = []
    ep_final_in_thresh = []
    ep_m2d_in_thresh_rates = []
    ep_m2d_final_in_thresh = []
    ep_m2d_final_axis_err = []
    ep_m2d_final_axial_off = []
    ep_m2d_final_radial_err = []
    ep_m2d_final_clearance = []
    ep_m2d_success_final_axis_err = []
    ep_m2d_success_final_axial_off = []
    ep_m2d_success_final_radial_err = []
    ep_m2d_success_final_clearance = []
    ep_m2d_final_window_success = []
    ep_m2d_final_window_in_thresh_rates = []
    ep_m2d_final_window_axis_err = []
    ep_m2d_final_window_axial_off = []
    ep_m2d_final_window_radial_err = []
    ep_m2d_final_window_clearance = []
    start = 0
    for end in end_indices:
        ep_pa = in_thresh_pa_np[start:end + 1]
        ep_max_holds_pa.append(_max_consecutive_run(ep_pa))
        ep_in_thresh_rates.append(float(ep_pa.mean()) if len(ep_pa) else 0.0)
        ep_final_in_thresh.append(bool(ep_pa[-1]) if len(ep_pa) else False)
        if in_thresh_full_np is not None:
            ep_full = in_thresh_full_np[start:end + 1]
            ep_max_holds_full.append(_max_consecutive_run(ep_full))
        if in_thresh_m2d_np is not None:
            ep_m2d = in_thresh_m2d_np[start:end + 1]
            ep_max_hold_m2d = _max_consecutive_run(ep_m2d)
            ep_m2d_success = ep_max_hold_m2d >= hold_n_steps
            ep_max_holds_m2d.append(ep_max_hold_m2d)
            ep_m2d_in_thresh_rates.append(float(ep_m2d.mean()) if len(ep_m2d) else 0.0)
            ep_m2d_final_in_thresh.append(bool(ep_m2d[-1]) if len(ep_m2d) else False)
            ep_m2d_final_axis_err.append(float(axis_err_np[end]))
            win_len = min(final_window_steps, len(ep_m2d))
            if win_len > 0:
                win_start = end - win_len + 1
                win_slice = slice(win_start, end + 1)
                ep_win = ep_m2d[-win_len:]
                ep_m2d_final_window_success.append(
                    bool(len(ep_m2d) >= final_window_steps and ep_win.all())
                )
                ep_m2d_final_window_in_thresh_rates.append(float(ep_win.mean()))
                ep_m2d_final_window_axis_err.append(
                    float(np.mean(axis_err_np[win_slice]))
                )
                if axial_flat is not None:
                    ep_m2d_final_window_axial_off.append(
                        float(np.mean(axial_flat[win_slice]))
                    )
                if radial_flat is not None:
                    ep_m2d_final_window_radial_err.append(
                        float(np.mean(radial_flat[win_slice]))
                    )
                if clr_flat is not None:
                    ep_m2d_final_window_clearance.append(
                        float(np.mean(clr_flat[win_slice]))
                    )
            else:
                ep_m2d_final_window_success.append(False)
                ep_m2d_final_window_in_thresh_rates.append(0.0)
            if axial_flat is not None:
                final_axial = float(axial_flat[end])
                ep_m2d_final_axial_off.append(final_axial)
                if ep_m2d_success:
                    ep_m2d_success_final_axial_off.append(final_axial)
            if radial_flat is not None:
                final_radial = float(radial_flat[end])
                ep_m2d_final_radial_err.append(final_radial)
                if ep_m2d_success:
                    ep_m2d_success_final_radial_err.append(final_radial)
            if clr_flat is not None:
                final_clearance = float(clr_flat[end])
                ep_m2d_final_clearance.append(final_clearance)
                if ep_m2d_success:
                    ep_m2d_success_final_clearance.append(final_clearance)
            if ep_m2d_success:
                ep_m2d_success_final_axis_err.append(float(axis_err_np[end]))
        start = end + 1

    hold_flags_pa = np.asarray([mh >= hold_n_steps for mh in ep_max_holds_pa], dtype=bool)
    out = {
        "hold_success_rate": float(hold_flags_pa.mean()) if len(hold_flags_pa) else 0.0,
        "max_hold_mean": float(np.mean(ep_max_holds_pa)) if ep_max_holds_pa else 0.0,
        "in_thresh_rate": float(np.mean(ep_in_thresh_rates)) if ep_in_thresh_rates else 0.0,
        "final_in_thresh_rate": float(np.mean(ep_final_in_thresh)) if ep_final_in_thresh else 0.0,
        "pos_err_mean": float(pos_err.mean()),
        "axis_err_mean": float(axis_err.mean()),
    }
    if in_thresh_full_np is not None:
        hold_flags_full = np.asarray(
            [mh >= hold_n_steps for mh in ep_max_holds_full], dtype=bool
        )
        out["hold_success_rate_with_clearance"] = (
            float(hold_flags_full.mean()) if len(hold_flags_full) else 0.0
        )
        out["max_hold_full_mean"] = (
            float(np.mean(ep_max_holds_full)) if ep_max_holds_full else 0.0
        )
        out["clearance_pass_rate"] = clearance_pass_rate
        out["clearance_threshold_used"] = float(clearance_threshold)
    if in_thresh_m2d_np is not None:
        hold_flags_m2d = np.asarray(
            [mh >= hold_n_steps for mh in ep_max_holds_m2d], dtype=bool
        )
        def _mean_or_nan(values):
            return float(np.mean(values)) if values else float("nan")

        out.update({
            "hold_success_rate_m2d": (
                float(hold_flags_m2d.mean()) if len(hold_flags_m2d) else 0.0
            ),
            "max_hold_m2d_mean": (
                float(np.mean(ep_max_holds_m2d)) if ep_max_holds_m2d else 0.0
            ),
            "m2d_in_thresh_rate": (
                float(np.mean(ep_m2d_in_thresh_rates)) if ep_m2d_in_thresh_rates else 0.0
            ),
            "m2d_final_in_thresh_rate": (
                float(np.mean(ep_m2d_final_in_thresh)) if ep_m2d_final_in_thresh else 0.0
            ),
            "m2d_axis_pass_rate": float(axis_ok_np.mean()),
            "m2d_axial_pass_rate": axial_pass_rate,
            "m2d_radial_pass_rate": radial_pass_rate,
            "m2d_final_axis_err_mean": _mean_or_nan(ep_m2d_final_axis_err),
            "final_window_steps": float(final_window_steps),
            "final_window_success_rate_m2d": (
                float(np.mean(ep_m2d_final_window_success))
                if ep_m2d_final_window_success else 0.0
            ),
            "final_window_in_thresh_rate_m2d": (
                float(np.mean(ep_m2d_final_window_in_thresh_rates))
                if ep_m2d_final_window_in_thresh_rates else 0.0
            ),
            "final_window_axis_err_mean_m2d": (
                _mean_or_nan(ep_m2d_final_window_axis_err)
            ),
            "m2d_success_final_axis_err_mean": (
                _mean_or_nan(ep_m2d_success_final_axis_err)
            ),
        })
        if axial_threshold is not None:
            out["axial_threshold_used"] = float(axial_threshold)
            out["axial_off_mean"] = float(axial_flat.mean())
            out["axial_off_p10"] = float(np.percentile(axial_flat, 10))
            out["axial_off_p90"] = float(np.percentile(axial_flat, 90))
            out["m2d_final_axial_off_mean"] = _mean_or_nan(ep_m2d_final_axial_off)
            out["final_window_axial_off_mean_m2d"] = (
                _mean_or_nan(ep_m2d_final_window_axial_off)
            )
            out["m2d_success_final_axial_off_mean"] = (
                _mean_or_nan(ep_m2d_success_final_axial_off)
            )
        if radial_threshold is not None:
            out["radial_threshold_used"] = float(radial_threshold)
            out["radial_err_mean"] = float(radial_flat.mean())
            out["radial_err_p10"] = float(np.percentile(radial_flat, 10))
            out["radial_err_p90"] = float(np.percentile(radial_flat, 90))
            out["m2d_final_radial_err_mean"] = _mean_or_nan(ep_m2d_final_radial_err)
            out["final_window_radial_err_mean_m2d"] = (
                _mean_or_nan(ep_m2d_final_window_radial_err)
            )
            out["m2d_success_final_radial_err_mean"] = (
                _mean_or_nan(ep_m2d_success_final_radial_err)
            )
        if clr_flat is not None:
            out["m2d_final_clearance_mean"] = _mean_or_nan(ep_m2d_final_clearance)
            out["final_window_clearance_mean_m2d"] = (
                _mean_or_nan(ep_m2d_final_window_clearance)
            )
            out["m2d_success_final_clearance_mean"] = (
                _mean_or_nan(ep_m2d_success_final_clearance)
            )
    return out


def summarize_clearance(dataset, thresholds=(-0.02, 0.0, 0.02, 0.05)):
    """从 dataset.info["min_clearance"] (mushroom flatten 后已用 mask 对齐) 算
    distribution metrics. dataset 没启用 M2c (info 里没有 clearance) 时返回
    所有字段为 nan / 0.

    Returns dict:
        clearance_mean / p10 / min
        clearance_rate_at_<th>       per-step rate (不是 per-episode hold).
    """
    flat = _extract_clearance_from_info(dataset)
    if flat is None or flat.size == 0:
        out = {"clearance_mean": float("nan"),
               "clearance_p10": float("nan"),
               "clearance_min": float("nan")}
        for th in thresholds:
            out[f"clearance_rate_at_{int(th*100):+d}cm"] = 0.0
        return out
    out = {
        "clearance_mean": float(flat.mean()),
        "clearance_p10": float(np.percentile(flat, 10)),
        "clearance_min": float(flat.min()),
    }
    for th in thresholds:
        out[f"clearance_rate_at_{int(th*100):+d}cm"] = float((flat >= th).mean())
    return out


def summarize_approach(dataset, mdp=None):
    """从 dataset.info 的 M2d 几何量算分布 metrics.

    返回 axial_dist / axial_off / radial_err 的 mean / p10 / p90. 如果没有
    M2d instrumentation (未启用 rew_axial_off/rew_radial/threshold/log), 返回 nan.
    """
    axial_dist = axial_off = radial_err = None
    if mdp is not None and hasattr(mdp, "_compute_approach_errors"):
        try:
            _, _, _, next_state, _, _ = dataset.parse(to="torch")
            approach = mdp._compute_approach_errors(next_state)
        except Exception:
            approach = None
        if approach is not None:
            axial_dist = approach["axial_dist"].detach().cpu().numpy().reshape(-1)
            axial_off = approach["axial_off"].detach().cpu().numpy().reshape(-1)
            radial_err = approach["radial_err"].detach().cpu().numpy().reshape(-1)
    if axial_off is None:
        axial_dist = _extract_info_array(dataset, "axial_dist")
        axial_off = _extract_info_array(dataset, "axial_off")
        radial_err = _extract_info_array(dataset, "radial_err")
    if axial_off is None or radial_err is None:
        return {
            "axial_dist_mean": float("nan"),
            "axial_dist_p10": float("nan"),
            "axial_dist_p90": float("nan"),
            "axial_off_mean": float("nan"),
            "axial_off_p10": float("nan"),
            "axial_off_p90": float("nan"),
            "radial_err_mean": float("nan"),
            "radial_err_p10": float("nan"),
            "radial_err_p90": float("nan"),
        }

    out = {
        "axial_off_mean": float(axial_off.mean()),
        "axial_off_p10": float(np.percentile(axial_off, 10)),
        "axial_off_p90": float(np.percentile(axial_off, 90)),
        "radial_err_mean": float(radial_err.mean()),
        "radial_err_p10": float(np.percentile(radial_err, 10)),
        "radial_err_p90": float(np.percentile(radial_err, 90)),
    }
    if axial_dist is None:
        out.update({
            "axial_dist_mean": float("nan"),
            "axial_dist_p10": float("nan"),
            "axial_dist_p90": float("nan"),
        })
    else:
        out.update({
            "axial_dist_mean": float(axial_dist.mean()),
            "axial_dist_p10": float(np.percentile(axial_dist, 10)),
            "axial_dist_p90": float(np.percentile(axial_dist, 90)),
        })
    return out


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
