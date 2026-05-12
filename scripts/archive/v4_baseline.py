"""Stage 3 v4 reward baseline — code snapshot for manual restoration.

**This file is NOT runnable.** It is the frozen source of v4 reward code as
it existed before removal on 2026-05-12.  After removal, the main repo no
longer contains `--stage3`, `_compute_stage3_*`, `set_stage3_epoch`, or
`compute_stage3_metrics`.  This snapshot is the canonical reference for
re-introducing the v4 baseline if a paper ablation later requires it.

WHY v4 WAS REMOVED
------------------
- v4 (V-shape depth + Phase A/B/C schedule) was the prior reward design.
  It was superseded by `geom_stage=insert` SAC v8 (advance + dwell + bad_entry
  + penetration), which trains clean insertion from the preaxis warm-start.
- v4 code was kept as a paper baseline until 2026-05-12, when the user
  decided no v4 ablation table is needed.  ~530 lines of dead code were
  removed from main; this file is what you'd paste back to restore.

WHAT THIS SNAPSHOT CONTAINS
---------------------------
1. The three env methods (`_compute_stage3_reward_components`,
   `_compute_stage3_reward`, `set_stage3_epoch`) — verbatim, with type hints.
2. The `__init__` kwarg defaults block (`stage3=False` + 14 stage3_* args).
3. The `__init__` body state-init block (~30 `_stage3_*` attribute writes
   + validation + `_w_*` peak weight stores).
4. The reward() dispatch branch (one elif).
5. The info-dict writes inside `_create_info_dictionary` (5 stage3 keys).
6. `compute_stage3_metrics(dataset, mdp, hold_n_steps)` from _eval_utils.py.
7. The argparse CLI surfaces from train_sac.py / eval_sac.py /
   visualize_policy.py (10 train_sac + 8 eval_sac + 7 visualize args).
8. The train_sac main-loop dispatch (`set_stage3_epoch`, m3 metric block,
   wandb stage3 keys, ckpt-selection metric).
9. Notes on shared symbols that stayed in main but were *renamed* after
   v4 removal — namely the peg-sample offsets, which are also used by
   geom_stage and were renamed `_stage3_peg_sample_offsets` →
   `_peg_sample_offsets` (likewise the kwarg + the DEFAULT_* constant).

RESTORATION RECIPE
------------------
1. Add the kwarg defaults to `DualArmPegHoleEnv.__init__` signature
   (section 2 below).  Insert near the existing reward-weight defaults.
2. Add the state-init block to `__init__` body (section 3).  Order matters
   for the validation `raise ValueError` calls.
3. Re-add the three methods to the class body (section 1).
4. In `reward()`, restore the `elif self._stage3: normal = self._compute_stage3_reward(next_obs)` branch (section 4).
5. In `_create_info_dictionary`, restore the `if self._stage3 and self._cached_d is not None: ... info["stage3_*"] = ...` block (section 5).
6. Rename `_peg_sample_offsets` back to `_stage3_peg_sample_offsets` (and the kwarg + DEFAULT_* constant) — or keep the geom-friendly name and have
   `__init__` write both attributes pointing to the same tuple.  The v4
   reward code itself reads `self._stage3_peg_sample_offsets`; geom code
   reads the same offsets through whichever name is current.
7. Restore the shared gates: `if self._stage3 or self._geom_stage is not None`
   at the obs-dim setup, axis_resid forcing, obs-box construction, and
   geometry-cache block (4 sites).  After removal these were narrowed to
   `if self._geom_stage is not None`; restore the `_stage3 or` term.
8. Add `compute_stage3_metrics` back to scripts/_eval_utils.py (section 6).
9. Add the argparse blocks (section 7) to train_sac.py, eval_sac.py, and
   visualize_policy.py.  Also restore the `forward_kwargs` tuple entries
   in each that list the stage3_* kwargs.
10. Restore the train_sac main-loop calls (section 8):
    `mdp.set_stage3_epoch(stage3_epoch)`, the `m3 = compute_stage3_metrics(...)`
    call, the wandb stage3_* keys, the ckpt-selection metric for
    `mdp._stage3`, and the per-epoch logger blocks (`stage3 v4: ...`,
    `stage3 depth (V-shape): ...`, `stage3 schedule: ...`, `stage3 weights @
    raw_epoch=...`, `stage3 eval (radial_max 严格): ...`).
11. Restore eval_sac / visualize_policy stage3 dispatch (section 9) —
    `env_kwargs["stage3"] = True`, set_stage3_epoch + phase mapping,
    the viz mask + overlay code.
12. (Optional) Restore archived scripts/archive/dryrun_stage3_v4.py and
    scripts/archive/sanity_eval_stage3.py to runnable form — they still
    exist in the archive but rely on the env methods above.

After paste-back, run `python -m py_compile envs/dual_arm_peg_hole_env.py
scripts/train_sac.py scripts/eval_sac.py scripts/visualize_policy.py
scripts/_eval_utils.py` to catch typos.  Verify with a short training run
(`--stage3 --rew_depth 5.0 ...`) that v4 path still emits the expected
`stage3 eval (radial_max 严格): insert_hold_rate=...` log line.

See also
--------
- memory feedback_bimanual_sac_failures.md — v1-v7 failure modes (textual)
- memory feedback_bimanual_stage3_traps.md — design lessons that survived
- scripts/archive/dryrun_stage3_v4.py — v4 reward dry-run (docstring
  documents the formula; non-runnable post-removal)
- scripts/archive/sanity_eval_stage3.py — d-sign convention validation
- git history before 2026-05-12 removal commit — fully reproducible source
"""

# ============================================================================
# SECTION 1 — env methods (paste into class DualArmPegHoleEnv)
# ============================================================================

def _compute_stage3_reward_components(self, next_obs):
    """Stage 3 reward components (v4 design — V-shape depth + radial_max + axis-gated).

    sign 约定: d > 0 = inserted (验证见 scripts/archive/sanity_eval_stage3.py).
    所有 radial 信号都用 ``radial_max`` (与 insert mask 同源, 杆身斜也可见);
    depth 用 V-shape (峰值在 d_target, 两侧线性掉, 防 overshoot 飞远).

    Components (signed, 已含正负号):
        r_depth          +w_depth · (d_target - |d - d_target|), clamp(min=-cap)
        r_axis           -w_axis · axis_gate(pos_err) · axis_err
        r_radial_base    -w_radial · radial_max                     (always-on)
        r_radial_close   -w_rad2_eff · sigmoid((d-d_g)/s_g) · radial_max
        r_pos_anchor     -w_pos_anchor_eff · pos_err                (Phase A 主, B anneal)
        r_radial_anchor  -w_radial_anchor_eff · radial_max          (Phase B 接管, C 保留)
        r_insert         +w_insert_eff · 1[d>d_ins ∧ radial_max<r_ins ∧ axis<a_ins]
        r_joint_limit    -w_joint_limit · joint_limit_norm
        r_action         -w_action · ||raw_action||²
        r_home           -w_home · home_norm

    success 不 absorbing (Rule 1: 边界 Q-cliff). insert bonus 是 per-step,
    episode 走满 horizon 让 dwell 累积. eval 用 hold-N over insert_mask.

    v3 → v4 关键差别 (failed run 复盘):
        - depth: clip plateau → V-shape, 防 d=0.5m 飞远 plateau 拿满 reward
        - radial: tip → radial_max, reward 跟 success 同源
        - anchor: 单 pos_err → Phase A pos_err + Phase B/C radial_max,
                  避免 inserted state 被 anchor 反向拉回 preinsert
    """
    joint_pos = next_obs[..., _AGENT_OBS_JOINT_POS]
    pos_err = self._last_pos_err
    axis_err = self._last_axis_err
    d = self._cached_d
    radial_max = self._cached_radial_max

    joint_limit_norm = self._compute_joint_limit_norm(joint_pos)
    action_sq = (self._last_raw_action ** 2).sum(dim=-1)
    home_norm = self._compute_home_norm(joint_pos)
    axis_gate = self._compute_axis_gate(pos_err)

    depth_signed = self._w_depth * (
        self._stage3_d_target - torch.abs(d - self._stage3_d_target)
    )
    r_depth = torch.clamp(depth_signed, min=-self._stage3_depth_penalty_cap)

    radial_gate = torch.sigmoid((d - self._stage3_d_g) / self._stage3_s_g)
    r_radial_base = -self._w_radial * radial_max
    r_radial_close = -self._stage3_w_rad2_eff * radial_gate * radial_max

    r_pos_anchor = -self._stage3_w_pos_anchor_eff * pos_err
    r_radial_anchor = -self._stage3_w_radial_anchor_eff * radial_max

    insert_mask = (
        (d > self._stage3_d_ins)
        & (radial_max < self._stage3_r_ins)
        & (axis_err < self._stage3_a_ins)
    ).to(d.dtype)
    r_insert = self._stage3_w_insert_eff * insert_mask

    r_axis = -self._w_axis * axis_gate * axis_err
    r_joint_limit = -self._w_joint_limit * joint_limit_norm
    r_action = -self._w_action * action_sq
    r_home = -self._w_home * home_norm

    return {
        "r_depth": r_depth, "r_axis": r_axis,
        "r_radial_base": r_radial_base, "r_radial_close": r_radial_close,
        "r_pos_anchor": r_pos_anchor, "r_radial_anchor": r_radial_anchor,
        "r_insert": r_insert,
        "r_joint_limit": r_joint_limit, "r_action": r_action, "r_home": r_home,
    }


def _compute_stage3_reward(self, next_obs):
    """Sum of stage3 reward components. 单一来源是 _compute_stage3_reward_components."""
    comps = self._compute_stage3_reward_components(next_obs)
    return sum(comps.values())


def set_stage3_epoch(self, epoch):
    """更新 Stage 3 reward 的 epoch-dependent 权重 (curriculum schedule).

    train loop 每个 epoch 起调一次. stage3=False 时 no-op.

    **传入的是 actor-relative epoch**: train_sac 在 critic warmup 期间会传 0,
    actor 解冻后才开始计数. 见 train_sac.py 主 loop 里
    ``critic_only_epochs`` 与 ``stage3_epoch = max(0, epoch - critic_only_epochs)``.

        Phase A (epoch < rad2_ramp_start):
            pos_anchor_eff=peak, radial_anchor_eff=0, rad2_eff=0, insert_eff=0
        Phase B (rad2_ramp_start ≤ epoch < rad2_ramp_end):
            t = (epoch - start) / (end - start)
            pos_anchor_eff   = (1-t) · peak
            radial_anchor_eff=    t  · peak
            rad2_eff         =    t  · peak
            insert_eff       = 0
        Phase C (epoch ≥ rad2_ramp_end):
            pos_anchor_eff=0, radial_anchor_eff=peak, rad2_eff=peak,
            insert_eff=peak (阶跃打开 dwell bonus)
    """
    if not self._stage3:
        return
    s, e = self._stage3_rad2_ramp_start, self._stage3_rad2_ramp_end
    if epoch < s:
        ramp_t = 0.0
    elif epoch < e:
        ramp_t = (epoch - s) / max(e - s, 1)
    else:
        ramp_t = 1.0
    insert_on = 1.0 if epoch >= e else 0.0
    self._stage3_w_rad2_eff = ramp_t * self._w_radial_close_peak
    self._stage3_w_radial_anchor_eff = ramp_t * self._w_radial_anchor_peak
    self._stage3_w_pos_anchor_eff = (1.0 - ramp_t) * self._w_pos_anchor_peak
    self._stage3_w_insert_eff = insert_on * self._w_insert_peak
    self._stage3_current_epoch = int(epoch)


# ============================================================================
# SECTION 2 — __init__ kwarg defaults (paste into signature)
# ============================================================================

INIT_KWARG_DEFAULTS = """
    stage3=False,
    # peak weights — 训练全程最大值, schedule scale 由 set_stage3_epoch 算
    rew_depth=5.0,
    rew_radial=1.0,
    rew_radial_close=3.0,
    rew_insert=1.0,
    rew_pos_anchor=0.15,
    rew_radial_anchor=0.3,
    # thresholds (V-shape depth)
    stage3_d_target=0.03,
    stage3_depth_penalty_cap=0.5,
    stage3_d_g=-0.02,
    stage3_s_g=0.01,
    stage3_d_ins=0.025,
    stage3_r_ins=0.005,
    stage3_a_ins=0.30,
    stage3_peg_sample_offsets=None,   # None → DEFAULT_STAGE3_PEG_SAMPLE_OFFSETS
    # epoch schedule (通过 set_stage3_epoch(epoch) 触发, actor-relative)
    stage3_rad2_ramp_start=30,
    stage3_rad2_ramp_end=60,
"""


# ============================================================================
# SECTION 3 — __init__ body state init (paste after geom_stage validation)
# ============================================================================

INIT_BODY_STATE = """
    self._stage3 = bool(stage3)

    # peak weights
    self._w_depth = float(rew_depth)
    self._w_radial = float(rew_radial)
    self._w_radial_close_peak = float(rew_radial_close)
    self._w_insert_peak = float(rew_insert)
    self._w_pos_anchor_peak = float(rew_pos_anchor)
    self._w_radial_anchor_peak = float(rew_radial_anchor)

    # epoch-dependent effective weights (initial Phase A values)
    self._stage3_w_rad2_eff = 0.0
    self._stage3_w_insert_eff = 0.0
    self._stage3_w_pos_anchor_eff = self._w_pos_anchor_peak
    self._stage3_w_radial_anchor_eff = 0.0
    self._stage3_current_epoch = 0

    # thresholds
    self._stage3_d_target = float(stage3_d_target)
    self._stage3_depth_penalty_cap = float(stage3_depth_penalty_cap)
    self._stage3_d_g = float(stage3_d_g)
    self._stage3_s_g = float(stage3_s_g)
    self._stage3_d_ins = float(stage3_d_ins)
    self._stage3_r_ins = float(stage3_r_ins)
    self._stage3_a_ins = float(stage3_a_ins)
    if self._stage3_depth_penalty_cap < 0.0:
        raise ValueError(
            f"stage3_depth_penalty_cap must >= 0, got {self._stage3_depth_penalty_cap}"
        )
    if self._stage3_s_g <= 0.0:
        raise ValueError(f"stage3_s_g must > 0, got {self._stage3_s_g}")

    # peg-sample offsets — also used by geom_stage path; after removal kept
    # as self._peg_sample_offsets (no _stage3_ prefix).  v4 reward reads the
    # same tuple via whichever attribute name is current.
    if stage3_peg_sample_offsets is None:
        offsets = DEFAULT_STAGE3_PEG_SAMPLE_OFFSETS
    else:
        offsets = tuple(float(x) for x in stage3_peg_sample_offsets)
        if any(o > 0.0 or o < -_PEG_HEIGHT for o in offsets):
            raise ValueError(
                f"stage3_peg_sample_offsets 元素必须 ∈ [-{_PEG_HEIGHT:.3f}, 0]; "
                f"got {offsets}"
            )
    self._stage3_peg_sample_offsets = offsets

    # schedule
    self._stage3_rad2_ramp_start = int(stage3_rad2_ramp_start)
    self._stage3_rad2_ramp_end = int(stage3_rad2_ramp_end)
    if self._stage3_rad2_ramp_end < self._stage3_rad2_ramp_start:
        raise ValueError(
            f"stage3_rad2_ramp_end ({self._stage3_rad2_ramp_end}) must >= "
            f"stage3_rad2_ramp_start ({self._stage3_rad2_ramp_start})"
        )

    # mutex with geom_stage
    if self._stage3 and self._geom_stage is not None:
        raise ValueError(
            "--stage3 与 geom_stage 互斥: 旧 Stage3 v4 和几何新路径不能同时启用"
        )
"""


# ============================================================================
# SECTION 4 — reward() dispatch (paste as elif branch)
# ============================================================================

REWARD_DISPATCH = """
def reward(self, obs, action, next_obs, absorbing):
    if self._geom_stage is not None:
        normal = self._compute_geom_reward(next_obs)
    elif self._stage3:
        normal = self._compute_stage3_reward(next_obs)
    else:
        normal = self._compute_normal_reward(next_obs)
    ...
"""


# ============================================================================
# SECTION 5 — info-dict writes inside _create_info_dictionary
# ============================================================================

INFO_WRITES = """
if self._stage3 and self._cached_d is not None:
    d = self._cached_d
    radial_max = self._cached_radial_max
    axis_err = self._cached_axis_err
    insert_mask = (
        (d > self._stage3_d_ins)
        & (radial_max < self._stage3_r_ins)
        & (axis_err < self._stage3_a_ins)
    )
    info["stage3_d"] = d.to(torch.float32)
    info["stage3_radial_max"] = radial_max.to(torch.float32)
    info["stage3_axis_err"] = axis_err.to(torch.float32)
    info["stage3_insert_mask"] = insert_mask.to(torch.float32)
    if self._last_pos_err is not None:
        info["stage3_pos_err"] = self._last_pos_err.to(torch.float32)
"""


# ============================================================================
# SECTION 6 — compute_stage3_metrics (paste into scripts/_eval_utils.py)
# ============================================================================

def compute_stage3_metrics(dataset, mdp, hold_n_steps):
    """Stage 3 specific eval metrics, 用于 best-insert ckpt 选择 + Phase A/B/C 监控.

    数据源是 env 在 _create_info_dictionary 写入 dataset.info.data 的:
        stage3_d           [N_steps]  depth (d>0 = inserted)
        stage3_radial_max  [N_steps]  peg 沿轴 4 采样点取 max 的径向距离
        stage3_axis_err    [N_steps]  1+cos(peg_axis, hole_axis)
        stage3_insert_mask [N_steps]  与 reward 的 1_insert 严格一致
        stage3_pos_err     [N_steps]  ||peg_tip - preinsert_target||, 用于 _at_d_max 聚合

    "_at_d_max" 三件套 (radial_max / axis_err / pos_err) 在 *episode 内 d 最大的那一步*
    取值. 防 "先精对齐再飞过去" 的策略蒙混过关.
    """
    import numpy as np
    import torch

    _, _, _, _, _, last = dataset.parse(to="torch")
    info = dataset.info.data
    required = ("stage3_d", "stage3_radial_max", "stage3_axis_err",
                "stage3_insert_mask", "stage3_pos_err")
    missing = [k for k in required if info.get(k) is None]
    if missing:
        raise KeyError(
            f"compute_stage3_metrics 需要 dataset.info.data 含 {required}, 缺 {missing}. "
            "确认 mdp._stage3=True 且 _create_info_dictionary 写入了这些字段."
        )

    def _to_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    d = _to_tensor(info["stage3_d"])
    radial_max = _to_tensor(info["stage3_radial_max"])
    axis_err = _to_tensor(info["stage3_axis_err"])
    pos_err = _to_tensor(info["stage3_pos_err"])
    insert_mask = _to_tensor(info["stage3_insert_mask"]).to(torch.bool)

    last_np = last.cpu().numpy().astype(bool)
    insert_np = insert_mask.cpu().numpy().astype(bool)
    d_np = d.cpu().numpy()
    radial_max_np = radial_max.cpu().numpy()
    axis_err_np = axis_err.cpu().numpy()
    pos_err_np = pos_err.cpu().numpy()

    end_indices = np.flatnonzero(last_np)
    ep_max_runs, ep_d_max = [], []
    ep_radial_max_min, ep_axis_at_d_max = [], []
    ep_radial_at_d_max, ep_pos_at_d_max = [], []
    start = 0
    for end in end_indices:
        ins_ep = insert_np[start:end + 1]
        max_run, cur = 0, 0
        for flag in ins_ep:
            cur = cur + 1 if flag else 0
            if cur > max_run:
                max_run = cur
        ep_max_runs.append(max_run)
        d_ep = d_np[start:end + 1]
        rad_ep = radial_max_np[start:end + 1]
        ax_ep = axis_err_np[start:end + 1]
        pos_ep = pos_err_np[start:end + 1]
        d_max_idx = int(d_ep.argmax())
        ep_d_max.append(float(d_ep[d_max_idx]))
        ep_radial_max_min.append(float(rad_ep.min()))
        ep_axis_at_d_max.append(float(ax_ep[d_max_idx]))
        ep_radial_at_d_max.append(float(rad_ep[d_max_idx]))
        ep_pos_at_d_max.append(float(pos_ep[d_max_idx]))
        start = end + 1

    insert_step_rate = float(insert_np.mean()) if insert_np.size else 0.0
    insert_hold_flags = np.asarray(
        [r >= hold_n_steps for r in ep_max_runs], dtype=bool
    )
    insert_hold_rate = (
        float(insert_hold_flags.mean()) if len(insert_hold_flags) else 0.0
    )
    _mean_or_zero = lambda xs: float(np.mean(xs)) if xs else 0.0
    return {
        "insert_step_rate": insert_step_rate,
        "insert_hold_rate": insert_hold_rate,
        "max_insert_run_mean": _mean_or_zero(ep_max_runs),
        "d_max_mean": _mean_or_zero(ep_d_max),
        "radial_max_min_mean": _mean_or_zero(ep_radial_max_min),
        "radial_max_at_d_max_mean": _mean_or_zero(ep_radial_at_d_max),
        "axis_err_at_d_max_mean": _mean_or_zero(ep_axis_at_d_max),
        "pos_err_at_d_max_mean": _mean_or_zero(ep_pos_at_d_max),
    }


# ============================================================================
# SECTION 7 — CLI argparse blocks (train_sac / eval_sac / visualize_policy)
# ============================================================================

TRAIN_SAC_CLI = """
# ──────────────── Stage 3 (insertion) ────────────────
p.add_argument("--stage3", action="store_true",
               help="启用 Stage 3 reward + obs 41 维 + epoch schedule. 隐含 "
                    "use_axis_resid_obs=True. 必配 --exclude_ee_from_physx_self_collision.")
p.add_argument("--rew_depth", type=float, default=None,
               help="Stage 3 V-shape depth peak. 默认 5.0")
p.add_argument("--rew_radial", type=float, default=None,
               help="Stage 3 -w_rad · radial_max, 默认 1.0")
p.add_argument("--rew_radial_close", type=float, default=None,
               help="Stage 3 depth-gated radial close. 默认 3.0")
p.add_argument("--rew_insert", type=float, default=None,
               help="Stage 3 per-step insert dwell bonus peak. 默认 1.0")
p.add_argument("--rew_pos_anchor", type=float, default=None,
               help="-w_pos_anchor·pos_err peak. 默认 0.15")
p.add_argument("--rew_radial_anchor", type=float, default=None,
               help="-w_radial_anchor·radial_max peak. 默认 0.3")
p.add_argument("--stage3_d_target", type=float, default=None)
p.add_argument("--stage3_depth_penalty_cap", type=float, default=None)
p.add_argument("--stage3_d_g", type=float, default=None)
p.add_argument("--stage3_s_g", type=float, default=None)
p.add_argument("--stage3_d_ins", type=float, default=None)
p.add_argument("--stage3_r_ins", type=float, default=None)
p.add_argument("--stage3_a_ins", type=float, default=None)
p.add_argument("--stage3_rad2_ramp_start", type=int, default=None)
p.add_argument("--stage3_rad2_ramp_end", type=int, default=None)

# forward_kwargs tuple additions:
#   "rew_depth", "rew_radial", "rew_radial_close", "rew_insert",
#   "rew_pos_anchor", "rew_radial_anchor",
#   "stage3_d_target", "stage3_depth_penalty_cap",
#   "stage3_d_g", "stage3_s_g",
#   "stage3_d_ins", "stage3_r_ins", "stage3_a_ins",
#   "stage3_rad2_ramp_start", "stage3_rad2_ramp_end",

# dispatch (after env_kwargs constructed):
#   if args.stage3:
#       env_kwargs["stage3"] = True
#       if not args.exclude_ee_from_physx_self_collision:
#           print("[WARN] --stage3 没配 --exclude_ee_from_physx_self_collision; ...")
#   if args.stage3 and args.geom_stage is not None:
#       raise ValueError("--stage3 与 --geom_stage 互斥")
"""

EVAL_SAC_CLI = """
# Stage 3 eval CLI block (mirrors train_sac, plus stage3_eval_phase):
p.add_argument("--stage3", action="store_true",
               help="Stage 3 ckpt 必须传, env 切到 41 维 obs.")
p.add_argument("--stage3_eval_phase", type=str, default="C",
               choices=("A", "B", "C"),
               help="Stage 3 eval 时把 set_stage3_epoch 设到哪个 phase. 默认 C")
p.add_argument("--stage3_d_target", type=float, default=None)
p.add_argument("--stage3_depth_penalty_cap", type=float, default=None)
p.add_argument("--stage3_d_g", type=float, default=None)
p.add_argument("--stage3_s_g", type=float, default=None)
p.add_argument("--stage3_d_ins", type=float, default=None)
p.add_argument("--stage3_r_ins", type=float, default=None)
p.add_argument("--stage3_a_ins", type=float, default=None)

# dispatch:
#   if args.stage3:
#       env_kwargs["stage3"] = True
#   ... after env constructed:
#   if args.stage3:
#       if args.stage3_eval_phase == "A": stage3_eval_epoch = 0
#       elif args.stage3_eval_phase == "B":
#           stage3_eval_epoch = (mdp._stage3_rad2_ramp_start + mdp._stage3_rad2_ramp_end) // 2
#       else:
#           stage3_eval_epoch = mdp._stage3_rad2_ramp_end + 1
#       mdp.set_stage3_epoch(stage3_eval_epoch)
"""

VISUALIZE_POLICY_CLI = """
# Visualize Stage 3 (same 8 args; uses Phase C at start):
p.add_argument("--stage3", action="store_true")
p.add_argument("--stage3_d_target", type=float, default=None)
p.add_argument("--stage3_depth_penalty_cap", type=float, default=None)
p.add_argument("--stage3_d_g", type=float, default=None)
p.add_argument("--stage3_s_g", type=float, default=None)
p.add_argument("--stage3_d_ins", type=float, default=None)
p.add_argument("--stage3_r_ins", type=float, default=None)
p.add_argument("--stage3_a_ins", type=float, default=None)

# dispatch:
#   if args.stage3:
#       env_kwargs["stage3"] = True
#       mdp.set_stage3_epoch(mdp._stage3_rad2_ramp_end + 1)   # Phase C
#       print(f"[VIZ STAGE 3] insert thresh: d_ins=..., r_ins=..., a_ins=...")
#
# rollout overlay (per step):
#   if args.stage3 and mdp._cached_d is not None:
#       d = mdp._cached_d
#       rmax = mdp._cached_radial_max
#       axis = mdp._cached_axis_err
#       insert_mask = (d > mdp._stage3_d_ins) & (rmax < mdp._stage3_r_ins) & (axis < mdp._stage3_a_ins)
#       # ... draw overlay text with d / r_max / axis vs thresholds.
"""


# ============================================================================
# SECTION 8 — train_sac main loop hooks
# ============================================================================

TRAIN_SAC_MAIN_LOOP = """
# After args / mdp construction, before loop:
if mdp._stage3:
    logger.info(f"stage3 v4: w_depth={mdp._w_depth:.2f} ...")
    logger.info(f"stage3 depth (V-shape): d_target=... a_ins=...")
    logger.info(f"stage3 schedule: Phase A [0,{mdp._stage3_rad2_ramp_start}) → B → C ...")

# Inside epoch loop (after compute actor_epoch = max(0, epoch - critic_only_epochs)):
stage3_epoch = actor_epoch
mdp.set_stage3_epoch(stage3_epoch)

# After core.evaluate / compute_hold_metrics:
m3 = (compute_stage3_metrics(dataset, mdp, args.hold_success_steps)
      if mdp._stage3 else None)

# ckpt selection (replaces best_hold for stage3):
if mdp._stage3:
    track_rate = m3['insert_hold_rate']
    track_score = m3['max_insert_run_mean']
    # save under 'best_insert' name instead of 'best_hold'

# Per-epoch print:
if mdp._stage3:
    logger.info(
        f"stage3 weights @ raw_epoch={epoch} stage3_epoch={stage3_epoch}: "
        f"w_pos_anchor_eff=... w_rad2_eff=... w_insert_eff=..."
    )
    logger.info(
        f"stage3 eval (radial_max 严格): "
        f"insert_step_rate=... insert_hold_rate=... max_insert_run_mean=... "
        f"d_max_mean=... radial_max_min_mean=... radial_max_at_d_max_mean=... "
        f"axis_err_at_d_max_mean=... pos_err_at_d_max_mean=..."
    )

# wandb metric block (if args.wandb and mdp._stage3):
wandb_log_dict.update({
    "stage3_raw_epoch": epoch,
    "stage3_epoch": stage3_epoch,
    "stage3_w_pos_anchor_eff": mdp._stage3_w_pos_anchor_eff,
    "stage3_w_radial_anchor_eff": mdp._stage3_w_radial_anchor_eff,
    "stage3_w_rad2_eff": mdp._stage3_w_rad2_eff,
    "stage3_w_insert_eff": mdp._stage3_w_insert_eff,
    "stage3_insert_step_rate": m3["insert_step_rate"],
    "stage3_insert_hold_rate": m3["insert_hold_rate"],
    "stage3_max_insert_run_mean": m3["max_insert_run_mean"],
    "stage3_d_max_mean": m3["d_max_mean"],
    "stage3_radial_max_min_mean": m3["radial_max_min_mean"],
    "stage3_radial_max_at_d_max_mean": m3["radial_max_at_d_max_mean"],
    "stage3_axis_err_at_d_max_mean": m3["axis_err_at_d_max_mean"],
    "stage3_pos_err_at_d_max_mean": m3["pos_err_at_d_max_mean"],
})
"""


# ============================================================================
# SECTION 9 — verified training command (v4 baseline reference run)
# ============================================================================

TRAIN_COMMAND = """
# Reference v4 baseline run (from README, pre-removal).  warm-start: Stage 2
# oneshot ckpt (results/checkpoints/saved/Stage2_preaxis_strict_refine_parent_*).
python scripts/train_sac.py \\
    --stage3 \\
    --exclude_ee_from_physx_self_collision \\
    --rew_depth 5.0 --rew_radial 1.0 --rew_radial_close 3.0 \\
    --rew_insert 1.0 --rew_pos_anchor 0.15 --rew_radial_anchor 0.3 \\
    --stage3_d_target 0.03 --stage3_depth_penalty_cap 0.5 \\
    --stage3_d_g -0.02 --stage3_s_g 0.01 \\
    --stage3_d_ins 0.025 --stage3_r_ins 0.005 --stage3_a_ins 0.30 \\
    --stage3_rad2_ramp_start 30 --stage3_rad2_ramp_end 60 \\
    --load_agent results/checkpoints/saved/Stage2_preaxis_strict_refine_parent_2026-05-11_10-38-18_best_hold.msh \\
    --n_epochs 200 --num_envs 16 --horizon 200 \\
    --n_steps_per_epoch 3200
"""
