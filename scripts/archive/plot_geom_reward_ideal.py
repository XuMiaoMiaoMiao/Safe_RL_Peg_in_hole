"""
画出 geom reward 在理想 policy 轨迹下的曲线, 用来诊断 reward 设计本身是否够形成
basin / 是否有 dead zone / 各分量梯度量级是否合理.

不依赖 IsaacSim — 纯 numpy 算 reward 公式. 跟 envs/dual_arm_peg_hole_env.py
_compute_geom_reward_components (line 1049-1109) 严格同源.

run:
    /home/miao/miniconda3/envs/safe_rl/bin/python scripts/archive/plot_geom_reward_ideal.py
output:
    /tmp/geom_reward_ideal.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- reward configs (与训练 CLI 同源) ---------------------------------------
PREAXIS_DEFAULT = dict(w_d=8.0,  w_rt=2.0, w_rm=5.0, w_axis=1.0, d_sat=0.12, radial_sat=0.10)
PREAXIS_STRICT  = dict(w_d=10.0, w_rt=6.0, w_rm=3.5, w_axis=1.2, d_sat=0.30, radial_sat=1.00)
INSERT_DEFAULT  = dict(w_d=5.0,  w_rt=0.0, w_rm=5.0, w_axis=1.0, d_sat=0.12, radial_sat=0.10)
SOFT_WELL = dict(w=0.8, sd=0.020, sr=0.015, sa=0.150)

D_TARGET_PREAXIS = -0.08      # preaxis 固定
D_TARGET_INSERT_END = +0.03   # insert ramp 终点


def r_geom(d, radial_tip, radial_max, axis_err, d_target, cfg, soft=False):
    d_err = np.abs(d - d_target)
    d_err_sat  = np.minimum(d_err, cfg["d_sat"])
    rt_sat     = np.minimum(radial_tip, cfg["radial_sat"])
    rm_sat     = np.minimum(radial_max, cfg["radial_sat"])
    r = (-cfg["w_d"]    * d_err_sat
         -cfg["w_rt"]   * rt_sat
         -cfg["w_rm"]   * rm_sat
         -cfg["w_axis"] * axis_err)
    if soft:
        r = r + SOFT_WELL["w"] * np.exp(
            - (d_err      / SOFT_WELL["sd"])**2
            - (radial_max / SOFT_WELL["sr"])**2
            - (axis_err   / SOFT_WELL["sa"])**2
        )
    return r


def components(d, radial_tip, radial_max, axis_err, d_target, cfg):
    d_err = np.abs(d - d_target)
    return dict(
        r_d  = -cfg["w_d"]    * np.minimum(d_err, cfg["d_sat"]),
        r_rt = -cfg["w_rt"]   * np.minimum(radial_tip, cfg["radial_sat"]),
        r_rm = -cfg["w_rm"]   * np.minimum(radial_max, cfg["radial_sat"]),
        r_ax = -cfg["w_axis"] * axis_err,
        r_soft = SOFT_WELL["w"] * np.exp(
            - (d_err      / SOFT_WELL["sd"])**2
            - (radial_max / SOFT_WELL["sr"])**2
            - (axis_err   / SOFT_WELL["sa"])**2
        ),
    )


def lin_ramp(t, t0, t1, v0, v1):
    """连续线性 ramp, t<=t0 → v0, t>=t1 → v1."""
    if t1 == t0:
        return v0 if t < t0 else v1
    f = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return v0 + f * (v1 - v0)


# --- 理想 preaxis episode ---------------------------------------------------
# step 0-50:  prepos approach — d: 0.05 → -0.08, radial: 0.15 → 0.02, axis 1.2 → 0.10
# step 50-100: preaxis refine — hold d=-0.08, radial 0.02 → 0.005, axis 0.10 → 0.01
# step 100-200: preaxis hold  — 全部 small (理想 dwell)
def preaxis_ideal_traj(T=200):
    t = np.arange(T)
    d, rt, rm, ax = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    for i in t:
        if i < 50:
            d[i]  = lin_ramp(i, 0, 50,  0.05,  D_TARGET_PREAXIS)
            rt[i] = lin_ramp(i, 0, 50,  0.15,  0.02)
            rm[i] = lin_ramp(i, 0, 50,  0.18,  0.025)
            ax[i] = lin_ramp(i, 0, 50,  1.20,  0.10)
        elif i < 100:
            d[i]  = D_TARGET_PREAXIS
            rt[i] = lin_ramp(i, 50, 100, 0.02,  0.003)
            rm[i] = lin_ramp(i, 50, 100, 0.025, 0.006)
            ax[i] = lin_ramp(i, 50, 100, 0.10,  0.010)
        else:
            d[i]  = D_TARGET_PREAXIS
            rt[i] = 0.003
            rm[i] = 0.006
            ax[i] = 0.010
    return t, d, rt, rm, ax


# --- 理想 insert episode (mid-ramp 假设 d_target = -0.02) -------------------
# step 0-30:  approach — d: 0 → -0.02 (跟随 ramped target)
# step 30-100: hold @ d_target with small radial/axis
def insert_ideal_traj(T=150, d_target=-0.02):
    t = np.arange(T)
    d, rt, rm, ax = np.zeros(T), np.zeros(T), np.zeros(T), np.zeros(T)
    for i in t:
        if i < 30:
            d[i]  = lin_ramp(i, 0, 30, 0.00, d_target)
            rt[i] = lin_ramp(i, 0, 30, 0.02, 0.005)
            rm[i] = lin_ramp(i, 0, 30, 0.025, 0.008)
            ax[i] = lin_ramp(i, 0, 30, 0.20, 0.015)
        else:
            d[i]  = d_target
            rt[i] = 0.005
            rm[i] = 0.008
            ax[i] = 0.015
    return t, d, rt, rm, ax


# === 画图 ===================================================================
fig, axes = plt.subplots(3, 2, figsize=(15, 13))

# ---- (0,0) ideal preaxis trajectory: 状态量 --------------------------------
t, d_tj, rt_tj, rm_tj, ax_tj = preaxis_ideal_traj(200)
ax = axes[0, 0]
ax.plot(t, d_tj, label="d (depth)", color="C0")
ax.plot(t, rm_tj, label="radial_max", color="C4")
ax.plot(t, rt_tj, label="radial_tip", color="C3", alpha=0.6)
ax.plot(t, ax_tj / 10, label="axis_err / 10", color="C5")  # scale 显示
ax.axhline(D_TARGET_PREAXIS, ls="--", color="C0", alpha=0.4)
ax.axvline(50, ls=":", color="k", alpha=0.3)
ax.axvline(100, ls=":", color="k", alpha=0.3)
ax.set_xlabel("episode step")
ax.set_ylabel("state")
ax.set_title("(A) preaxis ideal trajectory — state variables")
ax.legend(loc="upper right", fontsize=9)
ax.grid(alpha=0.3)
ax.text(25, 0.08, "approach", ha="center", fontsize=9, alpha=0.7)
ax.text(75, 0.08, "refine", ha="center", fontsize=9, alpha=0.7)
ax.text(150, 0.08, "hold (ideal dwell)", ha="center", fontsize=9, alpha=0.7)

# ---- (0,1) ideal preaxis trajectory: reward components (STRICT cfg) --------
cfg = PREAXIS_STRICT
comps = [components(d_tj[i], rt_tj[i], rm_tj[i], ax_tj[i], D_TARGET_PREAXIS, cfg) for i in range(len(t))]
r_d  = np.array([c["r_d"]  for c in comps])
r_rt = np.array([c["r_rt"] for c in comps])
r_rm = np.array([c["r_rm"] for c in comps])
r_ax = np.array([c["r_ax"] for c in comps])
r_soft = np.array([c["r_soft"] for c in comps])
r_dense = r_d + r_rt + r_rm + r_ax
r_with_soft = r_dense + r_soft

ax = axes[0, 1]
ax.plot(t, r_d,  label="r_d   (w=10)", color="C0")
ax.plot(t, r_rt, label="r_rt  (w=6)",  color="C3")
ax.plot(t, r_rm, label="r_rm  (w=3.5)", color="C4")
ax.plot(t, r_ax, label="r_axis (w=1.2)", color="C5")
ax.plot(t, r_dense, label="total dense", color="k", linewidth=2)
ax.plot(t, r_with_soft, label="total + soft well (w=0.8)", color="C2", linewidth=2, linestyle="--")
ax.axvline(50, ls=":", color="k", alpha=0.3)
ax.axvline(100, ls=":", color="k", alpha=0.3)
ax.set_xlabel("episode step")
ax.set_ylabel("per-step reward")
ax.set_title("(B) preaxis ideal — per-step reward (STRICT REFINE cfg)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

# ---- (1,0) r vs d slice (radial=0, axis=0.01) ------------------------------
d_grid = np.linspace(-0.20, 0.06, 300)
ax = axes[1, 0]
for label, cfg in [("preaxis default", PREAXIS_DEFAULT), ("preaxis STRICT", PREAXIS_STRICT)]:
    r = [r_geom(d, 0, 0, 0.01, D_TARGET_PREAXIS, cfg) for d in d_grid]
    ax.plot(d_grid, r, label=label)
r_soft_curve = [r_geom(d, 0, 0, 0.01, D_TARGET_PREAXIS, PREAXIS_STRICT, soft=True) for d in d_grid]
ax.plot(d_grid, r_soft_curve, label="STRICT + soft well", linestyle="--", color="C2")
ax.axvline(D_TARGET_PREAXIS, ls="--", color="k", alpha=0.4, label=f"d_target={D_TARGET_PREAXIS}")
ax.set_xlabel("d (m)")
ax.set_ylabel("per-step reward")
ax.set_title("(C) slice: r vs d  [radial=0, axis=0.01]")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ---- (1,1) r vs axis_err slice (d=d_target, radial=0) ---------------------
a_grid = np.linspace(0, 1.5, 300)
ax = axes[1, 1]
for label, cfg in [("preaxis default", PREAXIS_DEFAULT), ("preaxis STRICT", PREAXIS_STRICT)]:
    r = [r_geom(D_TARGET_PREAXIS, 0, 0, a, D_TARGET_PREAXIS, cfg) for a in a_grid]
    ax.plot(a_grid, r, label=label)
r_soft_curve = [r_geom(D_TARGET_PREAXIS, 0, 0, a, D_TARGET_PREAXIS, PREAXIS_STRICT, soft=True) for a in a_grid]
ax.plot(a_grid, r_soft_curve, label="STRICT + soft well", linestyle="--", color="C2")
ax.axvspan(0, 0.15, alpha=0.10, color="green", label="soft σ (axis)")
ax.set_xlabel("axis_err  (= 1 + cos(peg_axis, hole_axis))")
ax.set_ylabel("per-step reward")
ax.set_title("(D) slice: r vs axis_err  [d=d_target, radial=0]")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ---- (2,0) reward gradient (∂r/∂axis) along axis_err ----------------------
ax = axes[2, 0]
# 数值梯度
for label, cfg in [("preaxis default", PREAXIS_DEFAULT), ("preaxis STRICT", PREAXIS_STRICT)]:
    r = np.array([r_geom(D_TARGET_PREAXIS, 0, 0, a, D_TARGET_PREAXIS, cfg) for a in a_grid])
    grad = -np.gradient(r, a_grid)  # 取负, 显示"惩罚收紧力度"
    ax.plot(a_grid, grad, label=label)
r_soft_arr = np.array([r_geom(D_TARGET_PREAXIS, 0, 0, a, D_TARGET_PREAXIS, PREAXIS_STRICT, soft=True) for a in a_grid])
grad_soft = -np.gradient(r_soft_arr, a_grid)
ax.plot(a_grid, grad_soft, label="STRICT + soft", linestyle="--", color="C2")
ax.axvline(SOFT_WELL["sa"], ls=":", color="C2", alpha=0.5, label=f"soft σ_axis={SOFT_WELL['sa']}")
ax.axvline(SOFT_WELL["sa"]/np.sqrt(2), ls=":", color="C2", alpha=0.3, label="σ/√2 (soft grad peak)")
ax.set_xlabel("axis_err")
ax.set_ylabel("−∂r/∂axis  (pull toward 0)")
ax.set_title("(E) gradient pressure on axis  (higher = stronger pull)")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# ---- (2,1) insert mode: episode trajectory --------------------------------
t_ins, d_ins, rt_ins, rm_ins, ax_ins = insert_ideal_traj(T=150, d_target=-0.02)
cfg = PREAXIS_STRICT  # insert 用同一组 strict refine 权重做示意 (env default insert 是另一套)
comps_ins = [components(d_ins[i], rt_ins[i], rm_ins[i], ax_ins[i], -0.02, cfg) for i in range(len(t_ins))]
r_d_ins  = np.array([c["r_d"]  for c in comps_ins])
r_rt_ins = np.array([c["r_rt"] for c in comps_ins])
r_rm_ins = np.array([c["r_rm"] for c in comps_ins])
r_ax_ins = np.array([c["r_ax"] for c in comps_ins])
r_dense_ins = r_d_ins + r_rt_ins + r_rm_ins + r_ax_ins
r_soft_ins  = np.array([c["r_soft"] for c in comps_ins])
ax = axes[2, 1]
ax.plot(t_ins, r_d_ins,  label="r_d", color="C0")
ax.plot(t_ins, r_rt_ins, label="r_rt", color="C3")
ax.plot(t_ins, r_rm_ins, label="r_rm", color="C4")
ax.plot(t_ins, r_ax_ins, label="r_axis", color="C5")
ax.plot(t_ins, r_dense_ins, label="total dense", color="k", linewidth=2)
ax.plot(t_ins, r_dense_ins + r_soft_ins, label="+ soft well", color="C2", linewidth=2, linestyle="--")
ax.axvline(30, ls=":", color="k", alpha=0.3)
ax.set_xlabel("episode step")
ax.set_ylabel("per-step reward")
ax.set_title("(F) insert ideal (d_target=-0.02, mid-ramp) — per-step reward")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
out = "/tmp/geom_reward_ideal.png"
plt.savefig(out, dpi=120)
print(f"saved {out}")


# === 关键数值汇总 =============================================================
print()
print("=== 理想 preaxis dwell state (d=-0.08, radial=0.006, axis=0.01) ===")
state = dict(d=D_TARGET_PREAXIS, radial_tip=0.003, radial_max=0.006, axis_err=0.01)
for cname, cfg in [("default", PREAXIS_DEFAULT), ("STRICT", PREAXIS_STRICT)]:
    c = components(state["d"], state["radial_tip"], state["radial_max"], state["axis_err"], D_TARGET_PREAXIS, cfg)
    r_total = c["r_d"] + c["r_rt"] + c["r_rm"] + c["r_ax"]
    print(f"  {cname:8s}: r_d={c['r_d']:+.4f} r_rt={c['r_rt']:+.4f} r_rm={c['r_rm']:+.4f} r_ax={c['r_ax']:+.4f} | total={r_total:+.4f}")
    print(f"           soft well bonus @ this state: {c['r_soft']:+.4f}")
print()

print("=== preaxis 远场 (d=0.05, radial=0.15, axis=1.2) ===")
state = dict(d=0.05, radial_tip=0.15, radial_max=0.18, axis_err=1.2)
for cname, cfg in [("default", PREAXIS_DEFAULT), ("STRICT", PREAXIS_STRICT)]:
    c = components(state["d"], state["radial_tip"], state["radial_max"], state["axis_err"], D_TARGET_PREAXIS, cfg)
    r_total = c["r_d"] + c["r_rt"] + c["r_rm"] + c["r_ax"]
    print(f"  {cname:8s}: r_d={c['r_d']:+.4f} r_rt={c['r_rt']:+.4f} r_rm={c['r_rm']:+.4f} r_ax={c['r_ax']:+.4f} | total={r_total:+.4f}")
print()

print("=== 当前 ckpt 实测 mean state (d=-0.08, radial≈0.02-0.05, axis≈0.40) ===")
state = dict(d=D_TARGET_PREAXIS, radial_tip=0.03, radial_max=0.03, axis_err=0.40)
for cname, cfg in [("default", PREAXIS_DEFAULT), ("STRICT", PREAXIS_STRICT)]:
    c = components(state["d"], state["radial_tip"], state["radial_max"], state["axis_err"], D_TARGET_PREAXIS, cfg)
    r_total = c["r_d"] + c["r_rt"] + c["r_rm"] + c["r_ax"]
    print(f"  {cname:8s}: r_d={c['r_d']:+.4f} r_rt={c['r_rt']:+.4f} r_rm={c['r_rm']:+.4f} r_ax={c['r_ax']:+.4f} | total={r_total:+.4f}")
    print(f"           soft well bonus: {c['r_soft']:+.4e}  (≈0 时说明 state 在 soft basin 外)")
