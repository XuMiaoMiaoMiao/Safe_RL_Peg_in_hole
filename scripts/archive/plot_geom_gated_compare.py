"""
对比 additive vs alignment-gated progress reward 的局部 trade-off (codex 2026-05-11 提案).

目的: 验证 gated reward **结构性**改变了 stay-vs-advance 排序 — additive 下
agent 没有强 incentive 推进 (stay ≈ advance+bad), gated 下 advance+bad 显著
比 stay 差, 而 ideal advance 显著比两者好.

**这是设计验证工具, 不是 CLI 回归测试.** 配置写死在 ADDITIVE_INSERT /
GATED_INSERT / HYBRID_INSERT 三组里, 用来证明 codex 的局部排序假设数值上成立.
不能拿它来验证任意 CLI 配置 (例如改了 sigma / w_progress 后想看新效果, 请直接
改这文件里的 cfg dict 重跑, 不要假设它自动反映 env CLI 默认值).

不依赖 IsaacSim — 纯 numpy 算 reward 公式. 跟
envs/dual_arm_peg_hole_env.py::_compute_geom_reward_components (含 codex
gated progress 分支, progress_cap 用 d_target_eff) 严格同源.

run:
    /home/miao/miniconda3/envs/safe_rl/bin/python scripts/archive/plot_geom_gated_compare.py
output:
    /tmp/geom_gated_compare.png
    + 控制台 codex 3-state 比较表
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- configs (insert mode, w_d=0 since gated progress replaces axial pull) --
# 严格按 codex 提案: 保留 -w_radial, -w_axis (dense 稳定项), 删掉 r_geom_d
# (用 gated progress 做主推进信号).
ADDITIVE_INSERT = dict(
    w_d=5.0, w_rt=0.0, w_rm=5.0, w_axis=1.0,
    d_sat=0.30, radial_sat=1.0,
    w_progress=0.0,   # additive: no gated progress
    sigma_r=0.025, sigma_a=0.30,
)
GATED_INSERT = dict(
    w_d=0.0,          # codex 提案: 删掉 r_geom_d
    w_rt=0.0, w_rm=5.0, w_axis=1.0,
    d_sat=0.30, radial_sat=1.0,
    w_progress=8.0,   # codex 推荐 8-10
    sigma_r=0.025, sigma_a=0.30,
)
# 混合: r_geom_d 减半保留 (我之前提的弱 anchor, 防 peg 后退)
HYBRID_INSERT = dict(
    w_d=2.0, w_rt=0.0, w_rm=5.0, w_axis=1.0,
    d_sat=0.30, radial_sat=1.0,
    w_progress=8.0,
    sigma_r=0.025, sigma_a=0.30,
)

D_NEG = -0.08
D_POS = +0.03


def r_geom(d, radial_tip, radial_max, axis_err, d_target_eff, cfg):
    """跟 env._compute_geom_reward_components 同源.
    d_target_eff 是当前 ramp 值 (set_geom_epoch 设置的, prepos/preaxis 时 == d_neg).
    progress cap 是 d_target_eff - d_neg, 不是 d_pos - d_neg, 跟 env 修复后一致.
    """
    d_err = np.abs(d - d_target_eff)
    d_err_sat  = np.minimum(d_err, cfg["d_sat"])
    rt_sat     = np.minimum(radial_tip, cfg["radial_sat"])
    rm_sat     = np.minimum(radial_max, cfg["radial_sat"])
    r = (-cfg["w_d"]    * d_err_sat
         -cfg["w_rt"]   * rt_sat
         -cfg["w_rm"]   * rm_sat
         -cfg["w_axis"] * axis_err)
    if cfg["w_progress"] > 0.0:
        progress_range = D_POS - D_NEG
        progress_cap = max(0.0, min(progress_range, d_target_eff - D_NEG))
        progress = np.clip(d - D_NEG, 0.0, progress_cap)
        gate = np.exp(
            - (radial_max / cfg["sigma_r"])**2
            - (axis_err   / cfg["sigma_a"])**2
        )
        r = r + cfg["w_progress"] * progress * gate
    return r


def fmt(x):
    return f"{x:+.4f}" if isinstance(x, float) else str(x)


# === codex 的 3-state 比较 (insert mid-ramp, d_target=-0.02) ==================
print("=" * 86)
print("Codex 3-state local trade-off (d_target_eff=-0.02, insert ramp 中段)")
print("=" * 86)
states = [
    ("stay (preaxis, no progress)",        dict(d=-0.08, rt=0.005, rm=0.015, ax=0.05)),
    ("advance + degraded alignment",       dict(d=-0.02, rt=0.020, rm=0.030, ax=0.30)),
    ("ideal advance (aligned + progress)", dict(d=-0.02, rt=0.005, rm=0.015, ax=0.05)),
    ("far field (no preaxis ckpt)",        dict(d=+0.05, rt=0.15,  rm=0.18,  ax=1.2)),
]
d_target_for_calc = -0.02  # mid-ramp
print(f"{'state':<44} | {'additive':>10} | {'gated':>10} | {'hybrid':>10}")
print("-" * 86)
rows = []
for name, s in states:
    r_add = r_geom(s["d"], s["rt"], s["rm"], s["ax"], d_target_for_calc, ADDITIVE_INSERT)
    r_gat = r_geom(s["d"], s["rt"], s["rm"], s["ax"], d_target_for_calc, GATED_INSERT)
    r_hyb = r_geom(s["d"], s["rt"], s["rm"], s["ax"], d_target_for_calc, HYBRID_INSERT)
    print(f"{name:<44} | {r_add:>+10.4f} | {r_gat:>+10.4f} | {r_hyb:>+10.4f}")
    rows.append((name, r_add, r_gat, r_hyb))

# Δ 排序
print()
print("Δ (advance+bad − stay): 推进收益 — 应当 < 0 (推进且姿态烂不应比停留好)")
stay_add, adv_add, _ = rows[0][1], rows[1][1], None
stay_gat, adv_gat = rows[0][2], rows[1][2]
stay_hyb, adv_hyb = rows[0][3], rows[1][3]
print(f"  additive:  {adv_add - stay_add:+.4f}   ← codex 说的 '排序不强' (~0)")
print(f"  gated:     {adv_gat - stay_gat:+.4f}   ← 应当显著负, 强惩罚烂姿态推进")
print(f"  hybrid:    {adv_hyb - stay_hyb:+.4f}")
print()
print("Δ (ideal − stay): 正确推进的奖励 — 应当 > 0")
ideal_add, ideal_gat, ideal_hyb = rows[2][1], rows[2][2], rows[2][3]
print(f"  additive:  {ideal_add - stay_add:+.4f}")
print(f"  gated:     {ideal_gat - stay_gat:+.4f}")
print(f"  hybrid:    {ideal_hyb - stay_hyb:+.4f}")
print()


# === 画图 =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# ---- (0,0) reward vs d, alignment good (rm=0.015, ax=0.05) ----------------
d_grid = np.linspace(-0.12, 0.06, 300)
ax = axes[0, 0]
for label, cfg, color in [
    ("additive (w_d=5, w_progress=0)", ADDITIVE_INSERT, "C0"),
    ("gated    (w_d=0, w_progress=8)", GATED_INSERT,    "C2"),
    ("hybrid   (w_d=2, w_progress=8)", HYBRID_INSERT,   "C1"),
]:
    r = [r_geom(d, 0.005, 0.015, 0.05, d_target_for_calc, cfg) for d in d_grid]
    ax.plot(d_grid, r, label=label, color=color, linewidth=2)
ax.axvline(D_NEG, ls="--", color="k", alpha=0.3, label=f"d_neg ({D_NEG})")
ax.axvline(d_target_for_calc, ls=":", color="k", alpha=0.5, label=f"d_target ({d_target_for_calc})")
ax.axvline(D_POS, ls="--", color="k", alpha=0.3, label=f"d_pos (+{D_POS})")
ax.set_xlabel("d (axial position, m)")
ax.set_ylabel("per-step reward")
ax.set_title("(A) r vs d  [ALIGNED: rm=0.015, axis=0.05]\nshould prefer advance to d_pos")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

# ---- (0,1) reward vs d, alignment BAD (rm=0.030, ax=0.30) -----------------
ax = axes[0, 1]
for label, cfg, color in [
    ("additive", ADDITIVE_INSERT, "C0"),
    ("gated",    GATED_INSERT,    "C2"),
    ("hybrid",   HYBRID_INSERT,   "C1"),
]:
    r = [r_geom(d, 0.020, 0.030, 0.30, d_target_for_calc, cfg) for d in d_grid]
    ax.plot(d_grid, r, label=label, color=color, linewidth=2)
ax.axvline(D_NEG, ls="--", color="k", alpha=0.3)
ax.axvline(d_target_for_calc, ls=":", color="k", alpha=0.5)
ax.set_xlabel("d (m)")
ax.set_ylabel("per-step reward")
ax.set_title("(B) r vs d  [BAD ALIGNMENT: rm=0.030, axis=0.30]\nshould NOT prefer advancing (gate kills progress)")
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)

# ---- (1,0) bar chart: codex 3-state 局部对比 ------------------------------
ax = axes[1, 0]
labels = ["stay\n(preaxis)", "advance\n+bad align", "ideal\nadvance"]
x = np.arange(len(labels))
w = 0.27
adds = [rows[0][1], rows[1][1], rows[2][1]]
gats = [rows[0][2], rows[1][2], rows[2][2]]
hybs = [rows[0][3], rows[1][3], rows[2][3]]
ax.bar(x - w, adds, w, label="additive", color="C0")
ax.bar(x,     gats, w, label="gated",    color="C2")
ax.bar(x + w, hybs, w, label="hybrid",   color="C1")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("per-step reward")
ax.set_title("(C) local trade-off  [d_target=-0.02]\nadditive: stay ≈ advance+bad  (codex's complaint)")
ax.axhline(0, color="k", alpha=0.3, linewidth=0.5)
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
# 标注数值
for i, vals in enumerate([adds, gats, hybs]):
    offset = (i - 1) * w
    for j, v in enumerate(vals):
        ax.text(x[j] + offset, v + (0.02 if v < 0 else 0.02), f"{v:+.3f}",
                ha="center", fontsize=8, color="black")

# ---- (1,1) alignment gate shape ------------------------------------------
# 横轴 axis_err, 不同 rm 对应不同曲线
a_grid = np.linspace(0, 1.5, 300)
ax = axes[1, 1]
for rm_val, color in [(0.005, "C2"), (0.015, "C1"), (0.030, "C3"), (0.060, "C4")]:
    gate = np.exp(
        - (rm_val / GATED_INSERT["sigma_r"])**2
        - (a_grid / GATED_INSERT["sigma_a"])**2
    )
    ax.plot(a_grid, gate, label=f"radial_max={rm_val*100:.1f}cm", color=color, linewidth=2)
ax.axvline(GATED_INSERT["sigma_a"], ls=":", color="k", alpha=0.4, label=f"σ_a={GATED_INSERT['sigma_a']}")
ax.set_xlabel("axis_err")
ax.set_ylabel("alignment_gate  ∈ [0, 1]")
ax.set_title("(D) alignment_gate shape  (σ_r=0.025, σ_a=0.30)\n0=no progress reward, 1=full progress reward")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
out = "/tmp/geom_gated_compare.png"
plt.savefig(out, dpi=120)
print(f"saved {out}")
