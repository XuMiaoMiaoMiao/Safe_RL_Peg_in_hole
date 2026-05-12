"""双臂 peg-in-hole preinsert 任务 — mushroom-rl IsaacSim 向量化环境.

阶段 (stage flag 化, 同一个 env / 同一条 reward 骨架, reward 权重切换):
    Stage 1 = pos-only         rew_axis=0,   success_axis_threshold=inf
    Stage 2 = pos + axis 对齐  rew_axis=0.5, success_axis_threshold=0.50
                              + axis_gate_radius=0.40m + rew_pos_success
    Stage 3g = 真插入          geom_stage="insert" (d / radial_max / axis /
                              penetration-aware reward)

切换 stage 不改 env 结构, 只改 reward 权重和 success_axis_threshold. Stage 1
→ Stage 2 用 --load_agent + --actor_only_warmstart 做 actor 转移, obs 维度
保持一致 (推荐 --use_axis_resid_obs, 34 维; 默认 32 维 base 是历史 Stage 1 路径).

每臂控全部 7 DoF (A1-A7), 14 维 joint velocity 动作.

观测 (32 维 base / 34 维 axis_resid, 见 use_axis_resid_obs):
    joint_pos          (14) 左右臂 A1-A7 关节角
    joint_vel          (14) 左右臂 A1-A7 关节角速度
    pos_vec            (3)  peg_tip - preinsert_target
    axis_dot           (1)  dot(peg_axis, hole_axis) ∈ [-1, +1]   ← 32D base
                            -1 = 完美轴反平行 (理想对齐).
    axis_resid         (3)  peg_axis + hole_axis (world frame)    ← 34D, 推荐
                            模长 ∈ [0,2], 0=完美反对齐, 2=同向.
                            ||resid||²/2 = 1+dot = axis_err 同语义.

动作 (14):
    action ∈ [-1,1]^14 → joint velocity 指令 (rad/s), 系数 action_scale.

Reward (统一骨架):
    - w_pos          * pos_err                            # ||peg_tip - preinsert_target||
    - w_axis * gate  * axis_err                           # gate ∈ [0,1] 按距离门控, 远处不压
    - w_joint_limit  * joint_limit_norm
    - w_action       * sum(a_i^2)                         # raw action, 解耦 action_scale
    - w_home         * sum( ((q-q_home)/joint_range)^2 )  # 全 stage tie-breaker
    + w_pos_success  * 1[pos_err < pos_th]                # 防 Stage 1→Stage 2 success 断崖
    + w_success      * 1[full_success]                    # per-step dwell bonus, 不终止
    full_success = (pos_err < pos_th) ∧ (axis_err < axis_th)
                   # axis_th=inf 时退化为 pos-only — Stage 1 语义
    gate = clamp((axis_gate_radius - pos_err) / (axis_gate_radius - pos_th), 0, 1)
                   # axis_gate_radius=inf 时 gate ≡ 1 (不门控, Stage 1 / 旧行为)

终止:
    - 自碰撞 (双信号 OR, 任一触发即吸收 r = r_min / (1 - γ)):
        * PhysX 接触力 > collision_force_threshold
        * sphere-proxy clearance < clearance_hard (PhysX 在 1cm-5cm 边缘失明
          的几何兜底, 默认 clearance_hard=0.0 即球壳一接触就算碰撞)
    - success 本身不终止 (沿用 phase 1 结论, 避免 Q-target 边界断崖, 见
      feedback_bimanual_reward_shaping.md Rule 1)
    - hold-N (success 连续 N 步) 软 absorbing + terminal_hold_bonus 默认关闭
      (terminal_hold_bonus=0). 启用时 (传 >0) 会在 episode 收尾给 cliff,
      让 -w_pos*pos_err / -w_axis*axis_err 的连续梯度提前停止, 阈值边精度
      明显劣化; 当前默认走 horizon, 让精度信号跑满, hold_success_steps
      仅作 eval 指标 (compute_hold_metrics 的 hold_n_steps).

PEG/HOLE 几何 — 解析式 frame (不依赖 XFormPrim):
    peg/hole 是 EE link 下的 USD over. 当前 USD asset 已带 invisible collision
    proxy (peg Cylinder; hole Cube wall ring, CollisionAPI/PhysxCollisionAPI),
    但 Peg/Hole 本身故意没有 RigidBodyAPI / MassAPI: 它们是 EE link 的附属
    collision shapes, 不独立进入 articulation 质量矩阵. reward/obs 里的几何
    pose 完全由 EE link 的世界位姿 + 一个常量本地偏移解析计算:
        peg_tip_world  = LeftEE_pos  + R(LeftEE_quat)  · PEG_TIP_OFFSET_IN_LEFTEE
        peg_axis_world =                R(LeftEE_quat)  · PEG_AXIS_IN_LEFTEE
        hole_entry / hole_axis 同理 (RightEE).
    所以训练 headless 下完全不需要 XFormPrim/Fabric flush — 只要 BODY_POS +
    BODY_ROT 是 fresh 的, 帧就是对的. visualize_* 也走同一条解析路径,
    XFormPrim 不再使用.
"""

import math
from pathlib import Path

import torch

from mushroom_rl.environments import IsaacSim
from mushroom_rl.utils.isaac_sim import ObservationType, ActionType
from mushroom_rl.rl_utils.spaces import Box


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USD_PATH = (
    PROJECT_ROOT / "assets" / "usd" / "dual_arm_iiwa" / "dual_arm_iiwa_with_peghole.usda"
)

CONTROLLED_IDX = (1, 2, 3, 4, 5, 6, 7)  # A1-A7, 7 DoF/臂
LEFT_ARM_JOINTS = [f"left_arm_A{i}" for i in CONTROLLED_IDX]
RIGHT_ARM_JOINTS = [f"right_arm_A{i}" for i in CONTROLLED_IDX]
ARM_JOINTS = LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS  # 14

# Home pose — 取代 USD 自带的 zero 默认位姿. 双臂在胸前略弯肘的 ready 姿态,
# 视觉对称, 跟用户在 IsaacSim editor 里手动摆出来的形态一致.
#
# 注意: 这是 *起始* 姿态, 不是 *preinsert 目标* 姿态. reset 时 axis_dot 接近
# +1 (双臂 +Z 同向) 是完全正常的 — RL 的任务就是从同向起点学到反向对插
# (axis_dot ≈ -1). 不要为了让 reset 时 axis_dot 看起来好看而破坏 home 视觉.
HOME_JOINT_POS = (
    # left arm  A1     A2      A3      A4     A5   A6   A7
    -2.568,  -0.250, -0.078,  0.814,  0.0, 0.0,  0.010,
    # right arm — 镜像 (奇数 joint 取反, 偶数 joint 同号), 视觉对称
    +2.568,  -0.250, +0.078,  0.814,  0.0, 0.0, -0.010,
)
assert len(HOME_JOINT_POS) == 14, "HOME_JOINT_POS 必须 14 维 (左 7 + 右 7)"

LEFT_ARM_LINKS = [f"/left_arm_link_{i}" for i in range(1, 8)]
RIGHT_ARM_LINKS = [f"/right_arm_link_{i}" for i in range(1, 8)]

LEFT_EE_PATH = "/left_hande_robotiq_hande_link"
RIGHT_EE_PATH = "/right_hande_robotiq_hande_link"

LEFT_ARM_GROUP = LEFT_ARM_LINKS + [LEFT_EE_PATH]
RIGHT_ARM_GROUP = RIGHT_ARM_LINKS + [RIGHT_EE_PATH]

# ---- Sphere-proxy clearance 几何常量 -------------------------------------
# 把每条机械臂离散成一组球, 球心从 articulation BODY_POS 拿. 两侧球心两两算
# clearance = ||c_L - c_R|| - r_L - r_R, 取 min. 这是 PhysX 接触力检测之外的
# 几何 proxy, 用于阻止双臂 cross-over (PhysX 力检测在 1cm-5cm 边缘失明).
#
# 每侧 17 球 = 8 关节球 (link_0..link_7) + 7 中点球 (相邻 link 直线中点)
#            + 2 EE 球 (coupler, hande_link). finger 球去掉 — 它们在 peg/hole
#            视觉前面, 容易和零件重叠误触发, 而且 finger ↔ finger 不是 cross-over
#            关键区 (cross-over 主要发生在 link / hande_link 段).
# 中点用 0.5*(BODY_POS[i] + BODY_POS[i+1]); iiwa link 大体直筒, 直线中点
# 与 mesh 几何中心差距小, 不需要查 inertial / visual mesh, 完全只读 BODY_POS.
LEFT_ARM_JOINT_BODY_NAMES = [f"left_arm_link_{i}" for i in range(0, 8)]   # 8 球
RIGHT_ARM_JOINT_BODY_NAMES = [f"right_arm_link_{i}" for i in range(0, 8)]
LEFT_EE_PROXY_BODY_NAMES = [
    "left_hande_robotiq_hande_coupler",
    "left_hande_robotiq_hande_link",
]
RIGHT_EE_PROXY_BODY_NAMES = [
    "right_hande_robotiq_hande_coupler",
    "right_hande_robotiq_hande_link",
]
# 半径起步值: arm 6cm (link 直径 ~6-10cm 给 margin),
# ee 4cm (coupler + hande_link 整体宽度 ~6-8cm, 比起 finger 段 mass 更紧凑).
ARM_PROXY_RADIUS = 0.06
EE_PROXY_RADIUS = 0.04

# Peg/Hole 几何与挂载常量 — 必须与 build_peghole_usd.py 保持一致.
# 新约定 (Step 2 重新设计):
#   Peg/Hole 在 EE 帧里 = T(PART_X, 0, PART_Z), 不再 R_x(+90°).
#   零件 local +Z = EE local +Z = 夹爪正前方; peg/hole 直接从夹爪前方伸出.
#   peg_tip 在 Peg 局部 = (0, 0, +PEG_HEIGHT/2)
#   所以 peg_tip 在 LeftEE 帧:  (PART_X, 0, PART_Z + PEG_HEIGHT/2)
_PART_X = -0.0055
# PART_Z: peg/hole 局部坐标原点沿 EE +Z 的偏移. 0.155 让 peg/hole 后端坐在 finger
# 末端 (~z=0.12), finger 只夹住零件最后端 ~3cm, 前段全部伸出.
_PART_Z = 0.155
# 2× 放大 (Step 2 重设计). hole_outer = 24mm 半径 = 48mm 直径, 给 Robotiq
# Hande 50mm 开度留 2mm 安全余量. 长宽等比放大保持原始比例.
_PEG_HEIGHT = 0.070
# USD-verified geometry for penetration metric (codex 2026-05-11 议:
# trajectory-level 几何约束, 不依赖 PhysX contact). 跟
# assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda 对齐:
# - Peg = Cylinder radius=0.016 height=0.07 (peg 实体半径)
# - Hole = 16 个 Cube wall ring, 内壁半径 0.020 (inner_r), 高 0.06 (沿 hole 轴)
# - Bottom stopper Cylinder radius=0.020 在 hole 底 (z=-0.0315 局部)
# 物理 clearance = 0.020 - 0.016 = 0.004m (peg shaft 表面到 hole 内壁的最小净空)
_PEG_RADIUS = 0.016
_HOLE_INNER_RADIUS = 0.020
_HOLE_OUTER_RADIUS = 0.024
_HOLE_WALL_THICKNESS = _HOLE_OUTER_RADIUS - _HOLE_INNER_RADIUS   # 0.004 = 4mm
_HOLE_DEPTH = 0.060
# axial 范围 (hole_entry 帧, hole_axis 指 OUT of hole):
# - 0 = hole entrance plane
# - -HOLE_DEPTH = hole bottom plane
# 一个 sample 真正"跟 hole wall 物理相交"需要双重条件:
#   1. axial_s ∈ [-_HOLE_DEPTH, 0]  (在 wall 的轴向范围内)
#   2. radial_s < _HOLE_OUTER_RADIUS + _PEG_RADIUS  (peg cross-section 跟 wall ring
#      有重叠的可能; 远场 peg 即使 axial 落入范围也不算穿模)
# penetration 物理上限 = wall_thickness = 4mm (peg 完全穿过 4mm 厚的 wall 后,
# peg 已在 wall 外侧空间, 不再"in wall"). 用 clamp(0, 4mm) 防 1.3m 这种远场假数据.
_HOLE_HEIGHT = 0.060
_PEG_TIP_LOCAL_Z = 0.5 * _PEG_HEIGHT       # 0.035
_HOLE_ENTRY_LOCAL_Z = 0.5 * _HOLE_HEIGHT   # 0.030

PEG_TIP_OFFSET_IN_LEFTEE = (_PART_X, 0.0, _PART_Z + _PEG_TIP_LOCAL_Z)
HOLE_ENTRY_OFFSET_IN_RIGHTEE = (_PART_X, 0.0, _PART_Z + _HOLE_ENTRY_LOCAL_Z)

# peg_axis / hole_axis: 都沿 EE 局部 +Z (夹爪前方). 训练目标要求两者在 world
# frame 中反平行: axis_dot = -1, axis_err = 1 + dot = 0. 当前 HOME reset 下
# axis_err 接近 2 是正常的; Stage 2 才负责从同向附近学到反向对齐.
PEG_AXIS_IN_LEFTEE = (0.0, 0.0, +1.0)
HOLE_AXIS_IN_RIGHTEE = (0.0, 0.0, +1.0)

# peg_axis_quat / hole_axis_quat — 给 visualize_targets 的箭头 orient 用.
# 现在轴方向 = EE +Z, 直接 apply LeftEE_quat 即可, offset = identity quat.
PEG_AXIS_QUAT_OFFSET = (1.0, 0.0, 0.0, 0.0)
HOLE_AXIS_QUAT_OFFSET = (1.0, 0.0, 0.0, 0.0)

# Agent obs 索引切片 — reward / is_absorbing 直接按位读, 不再走 obs_helper.
# 两种布局:
#   BASE (32 维):       joint_pos + joint_vel + pos_vec + axis_dot
#   AXIS_RESID (34 维): joint_pos + joint_vel + pos_vec + axis_resid
#                       axis_resid[3] = peg_axis + hole_axis (world frame).
#                       ||axis_resid|| = 2·|cos(θ_to_target/2)|, 0 = 完美反对齐,
#                       2 = 同向 (home pose). 全程光滑, 无 SO(3) ±π 奇异.
_AGENT_OBS_JOINT_POS = slice(0, 14)
_AGENT_OBS_JOINT_VEL = slice(14, 28)
_AGENT_OBS_POS_VEC = slice(28, 31)
_AGENT_OBS_AXIS_DOT = slice(31, 32)         # 仅 BASE
_AGENT_OBS_AXIS_RESID = slice(31, 34)       # 仅 AXIS_RESID (替换 axis_dot)
# Geom obs 在 AXIS_RESID 末尾追加 7 维 几何信号 (1+3+3):
#   d            = -axial_dist  (peg 进入 hole 深度标量, d>0 = inserted)
#   radial_vec   = (peg_tip - hole_entry) - axial_dist · hole_axis
#                  ↑ 世界帧 3D 向量, 但已减掉 hole_axis 投影 → axial 分量 = 0
#                    没做严格 hole-local frame rotation (省了构造正交 basis 的开销),
#                    actor 可以靠 axis_resid 推 hole_axis 方向, joint state 推世界
#                    朝向, 信息上等价于 hole-local 表示.
#   peg_in_hole  = peg_tip - hole_entry  (世界帧 3D 相对位移, 含 axial+radial 全分量)
_AGENT_OBS_STAGE3_D = slice(34, 35)
_AGENT_OBS_STAGE3_RADIAL_VEC = slice(35, 38)
_AGENT_OBS_STAGE3_PEG_IN_HOLE = slice(38, 41)
AGENT_OBS_DIM_BASE = 32
AGENT_OBS_DIM_AXIS_RESID = 34
AGENT_OBS_DIM_GEOM = 41

# Peg 沿轴采样偏移 (m, 从 tip 往 peg base 方向). peg 总长 7cm
# (_PEG_HEIGHT=0.070), 不要超 -0.06 否则采到 EE coupler 段虚拟杆.
# 用于 geom_stage radial_max / penetration_max 计算.
DEFAULT_PEG_SAMPLE_OFFSETS = (0.0, -0.02, -0.04, -0.06)

DEFAULT_PREINSERT_OFFSET = 0.05
DEFAULT_HOME_WEIGHTS = (1.0,) * 14


class DualArmPegHoleEnv(IsaacSim):
    def __init__(
        self,
        num_envs=2,
        horizon=150,
        gamma=0.99,
        headless=True,
        device="cuda:0",
        action_scale=0.4,
        initial_joint_noise=0.1,
        collision_force_threshold=10.0,
        reward_absorbing_r_min=-2.0,
        reward_scale=1.0,
        rew_pos=1.0,
        rew_axis=0.0,
        rew_success=2.0,
        rew_pos_success=0.0,
        rew_joint_limit=0.02,
        rew_action=0.005,
        rew_home=0.0,
        home_weights=None,
        success_hold_steps=10,
        terminal_hold_bonus=0.0,
        preinsert_success_pos_threshold=0.10,
        success_axis_threshold=float("inf"),
        # axis-gate: 把 axis 惩罚按距离门控, 远处不干扰位置学习,
        #   靠近 preinsert 才打开. inf = 不门控 (全 stage 一直施压).
        #   推荐 Stage 2 用 0.40m: 离 preinsert 40cm 起开始线性 ramp,
        #   到 pos_th=0.10m 时 gate 满.
        axis_gate_radius=float("inf"),
        joint_limit_margin_frac=0.8,
        preinsert_offset=DEFAULT_PREINSERT_OFFSET,
        # Sphere-proxy 自碰撞兜底 (PhysX 失明区补丁, 全 stage 通用):
        #   min_clearance < clearance_hard 时与 PhysX 力 OR, 触发 hard absorbing.
        #   default 0.0 = 球壳一接触就算碰撞. 设 -inf 即关闭.
        clearance_hard=0.0,
        proxy_arm_radius=ARM_PROXY_RADIUS,
        proxy_ee_radius=EE_PROXY_RADIUS,
        # Stage 3 里 peg/hole 有真实 collider, 且挂在左右 EE link 下。若继续把
        # EE link 放进 PhysX self-collision group, 正常 peg-hole 接触会被 arm_L
        # vs arm_R hard absorbing 误杀。打开此开关后, PhysX self-collision 只看
        # iiwa arm links; EE 区域仍由 sphere-proxy 几何兜底负责。
        exclude_ee_from_physx_self_collision=False,
        # obs 模式:
        #   use_axis_resid_obs=True → 34 维 (peg_axis+hole_axis 求和替换 axis_dot,
        #                              全程光滑无奇异, 当前 Stage 2 推荐)
        #   False                   → 32 维 base (axis_dot 标量, Stage 1 默认)
        use_axis_resid_obs=False,
        # Peg-shaft sample offsets — used by geom_stage radial_max /
        # penetration_max sampling.  None → DEFAULT_PEG_SAMPLE_OFFSETS.
        peg_sample_offsets=None,
        # ──────────────── Geometric preinsert (Stage 1g/2g/3g) ────────────────
        # 几何同源 reward 主线. obs 41D, 不用球形 pos_err.
        # geom_stage:
        #   None       → 旧 Stage 1/2 _compute_normal_reward (Lagrangian baseline 用)
        #   "prepos"   → Stage 1g: -w_geom_d·|d-d_target| - w_geom_radial_tip·sat(rad_tip)
        #   "preaxis"  → Stage 2g: + radial_max + axis dense (warm-start prepos ckpt)
        #   "insert"   → Stage 3g: 同 preaxis, d_target 由 set_geom_epoch 从 d_target_neg
        #                          线性 ramp 到 d_target_pos (insert 推进)
        # 不带 hard success cliff; success 仅写入 info dict 给 eval / best_hold ckpt.
        geom_stage=None,
        # d_target schedule (insert mode only). prepos / preaxis 用 d_target_neg 不动.
        geom_d_target_neg=-0.08,
        geom_d_target_pos=+0.03,
        geom_d_target_ramp_start=0,
        geom_d_target_ramp_end=60,
        # peak weights (per-step, 单位 m^-1 或无量纲, dense gradient only)
        rew_geom_d=None,
        rew_geom_radial_tip=None,
        rew_geom_radial_max=None,
        rew_geom_axis=None,
        # saturation (clip 远场幅值, 防 cold start critic 难学)
        geom_d_sat=0.12,
        geom_radial_sat=0.10,
        # optional Gaussian soft success well (default off — codex 拍板默认 0)
        rew_geom_soft_success=0.0,
        geom_soft_d_sigma=0.02,
        geom_soft_radial_sigma=0.015,
        geom_soft_axis_sigma=0.30,
        # 可选: 把 penetration 纳入 soft well exponent (codex 2026-05-11 v4
        # "clean_dwell" 提案). default None = 不纳入 (向后兼容 3-项 well).
        # finite>0 = 加入第 4 项 exp(-(penetration/σ_pen)²), 让 well 只在
        # "干净 dwell" 时给正奖励, 穿模 dwell 直接拿 ~0. 推荐 0.001 = 1mm.
        geom_soft_penetration_sigma=None,
        # bootstrap success thresholds (训练期 best_hold ckpt 用;
        # 严格 eval 由 CLI 覆盖, 跑 strict eval 单独传更紧的值)
        geom_d_th=0.03,
        geom_r_tip_th=0.03,
        geom_r_max_th=0.03,
        geom_axis_th=0.40,
        # insert mode 真 insert 阈值 (geom_stage="insert" 时取代 d_th)
        geom_insert_d_ins=0.025,
        geom_insert_r_max_th=0.025,    # bootstrap (strict eval 用 0.015)
        # penetration 阈值 (codex 2026-05-11 v4): insert_mask 现加
        # penetration_max < geom_pen_th. 否则 best_hold ckpt 会被穿模状态
        # 误选 ("满 mask 但物理穿过 4mm wall" 假好). 默认 0.001 = 1mm 数值噪声
        # 容忍; 严格 eval 用 0.0005.
        geom_pen_th=0.001,
        # ─── Alignment-gated progress reward (codex 提案, 解决 additive 不强制
        # joint 满足问题). r_progress = w_progress · clamp(d - d_target_neg,
        # 0, d_target_pos - d_target_neg) · exp(-(rm/σ_r)² - (axis/σ_a)²).
        # w_progress=0 (默认) 关闭, 与原 additive 行为兼容. 推荐 insert 配方:
        # w_progress=8-10, σ_r=0.025 (= r_max_th), σ_a=0.30 (= axis_th).
        rew_geom_progress=None,
        geom_gate_radial_sigma=0.025,
        geom_gate_axis_sigma=0.30,
        # ─── Penetration-aware reward (SAC vs Lagrangian SAC 对比所需).
        # penetration_max 已经在 env 里算 (per-step physical 穿模量, [0, 4mm]).
        # 这里加 reward 接入 + cost signal switch:
        # - rew_geom_penetration: 软 penalty -w_pen·penetration_max. 默认 0=关.
        #   推荐 SAC: 10-20. Lagrangian SAC: 0 (用 cost 信号代替).
        # - geom_gate_penetration_sigma: gate 里 penetration 项的 σ. 设 finite 值
        #   (e.g. 0.002 = 2mm) 把 penetration 纳入 alignment_gate 乘子, 让 progress
        #   reward 在穿模时趋 0. 默认 None=不纳入 gate.
        # - cost_signal: 'collision' (老的 0/1 indicator) 或 'penetration'
        #   (= penetration_max 连续 [0, 4mm]). Lagrangian SAC 配 'penetration'.
        rew_geom_penetration=None,
        geom_gate_penetration_sigma=None,
        cost_signal="collision",
        # progress floor (codex 2026-05-11 v2): progress reward 起点.
        # 默认 0.0 = 只奖励 peg 真正越过 hole 入口 (d > 0). 旧行为 (奖励
        # "从 preinsert 接近 entrance") 用 -0.08 (= d_target_neg).
        # 改默认到 0.0 是因为旧公式让 agent 学到 "hover at entrance" 局部最优,
        # 不真插入也能拿 70%+ progress reward. 见 SAC_pen_aware_v1 失败分析.
        geom_progress_floor=0.0,
        # ─── Delta-progress (potential-based) reward (codex 2026-05-11 v3 提案).
        # 旧 state-based progress 让 agent "占位赚钱", 改成 Δphi 让 agent 必须
        # "前进才赚钱". phi(s) = clean_gate(s) × clamp((d - d_neg)/(d_pos - d_neg), 0, 1).
        # r_advance = w_advance × (phi_t - phi_{t-1}). 关键工程点: phi_prev 在
        # episode reset 时必须无效化, 否则跨 episode delta 会有 spurious 大值.
        # 默认 0 = off (向后兼容).
        rew_geom_advance=None,
        # ─── Task-ordering 修复 (codex 2026-05-11 v6 提案): clean insertion 是
        # 有顺序约束的任务 (先对齐再越过 entrance), 不是几项 reward 简单相加.
        # 之前所有 SAC 失败 (v1-v5) 共同模式: agent 找到"d 满足但 radial/axis 不
        # 满足"的捷径状态 (v5 ep37: d=+0.048 但 radial=16.8cm).
        # 两个 reward 结构修法:
        #
        # 1) geom_d_gate_mode: r_d 是否乘 alignment_gate, 让 r_d 只在 aligned
        #    状态有奖励. 默认 "off" (向后兼容); "alignment" 启用 codex 修法.
        #
        # 2) rew_geom_bad_entry: 显式惩罚 "d > 0 但对齐不达标". 当前公式
        #    使用归一化 violation 并 clamp 上界:
        #    depth_norm = clamp(d / d_target_pos, 0, 2)
        #    violation_i = clamp(metric_i / safe_i - 1, 0, 3)
        #    r_bad_entry = -w_be × depth_norm × clamp(sum_i violation_i, 0, 3)
        #    d ≤ 0 (peg 在 hole 外): 项 = 0
        #    d > 0 + 对齐 OK: 项 = 0
        #    d > 0 + 对齐烂: 越深越罚, 越烂越罚
        #    强制 task ordering: agent 必须在 d<0 阶段把对齐做对, 才敢越过 entrance.
        #    默认 0 = off.
        geom_d_gate_mode="off",
        rew_geom_bad_entry=None,
        geom_bad_entry_radial_safe=0.010,    # 10mm — 比 r_max_th 严
        geom_bad_entry_axis_safe=0.10,        # ~26° — 比 axis_th 严
        geom_bad_entry_pen_safe=0.0005,       # 0.5mm — 比 pen_th 严
        usd_path=None,
    ):
        self._action_scale = action_scale
        self._initial_joint_noise = initial_joint_noise
        self._collision_threshold = collision_force_threshold
        self._r_min = reward_absorbing_r_min
        self._reward_scale = reward_scale
        self._w_pos = rew_pos
        # rew_axis 默认 0 = Stage 1 行为 (axis 项消失). Stage 2 通过 CLI 打开.
        self._w_axis = rew_axis
        self._w_success = rew_success
        # pos-only success bonus: 维持 Stage 1 已学会的"进 pos 阈值给 +bonus"信号,
        # 避免 Stage 2 加 axis 后, Stage 1 成功状态突然失去 success bonus 造成断崖.
        # full_success (pos ∧ axis) 的 bonus 仍走 _w_success.
        self._w_pos_success = float(rew_pos_success)
        self._w_joint_limit = rew_joint_limit
        self._w_action = rew_action
        # home regularizer: -w_home · Σ_i home_weight_i ·
        # ((q_i - q_home_i) / joint_range_i)^2. 默认全 1, 保持旧行为.
        self._w_home = float(rew_home)
        if home_weights is None:
            self._home_weights_values = DEFAULT_HOME_WEIGHTS
        else:
            weights = tuple(float(w) for w in home_weights)
            if len(weights) == 7:
                weights = weights + weights
            if len(weights) != len(HOME_JOINT_POS):
                raise ValueError(
                    "home_weights 必须是 7 维(单臂, 自动复制到左右臂)或 14 维; "
                    f"传入 {len(weights)} 维: {weights}"
                )
            bad = [i for i, w in enumerate(weights) if not math.isfinite(w) or w < 0.0]
            if bad:
                raise ValueError(
                    f"home_weights 必须是有限非负数; 非法索引 {bad}, weights={weights}"
                )
            self._home_weights_values = weights
        # hold-N absorbing 设计 (沿用 phase 1): 连续 N 步在阈内即终止 + bonus.
        # bonus=0 时整个机制关闭 (baseline 行为).
        self._success_hold_steps = int(success_hold_steps)
        self._terminal_hold_bonus = float(terminal_hold_bonus)
        self._absorbing_terminal_active = self._terminal_hold_bonus > 0.0
        self._preinsert_success_pos_threshold = float(preinsert_success_pos_threshold)
        # success_axis_threshold 默认 inf = success 不检查 axis (Stage 1 行为).
        # Stage 2 训练阈值 0.50 (严格 eval 可降到 0.30/0.20). 用 inf 而不是 None
        # 让 success_mask 表达式不需要 None-check 分支, 永远是干净的
        # (pos<pos_th) & (axis_err<axis_th).
        self._success_axis_threshold = float(success_axis_threshold)
        self._joint_limit_margin_frac = joint_limit_margin_frac
        self._preinsert_offset = float(preinsert_offset)
        # axis_gate_radius: 距离阈值, axis 惩罚在 [pos_th, gate_radius] 区间线性
        # 从 0 ramp 到 1, 区间外 clamp. inf = 不门控 (向后兼容).
        self._axis_gate_radius = float(axis_gate_radius)
        if (math.isfinite(self._axis_gate_radius)
                and self._axis_gate_radius <= float(preinsert_success_pos_threshold)):
            raise ValueError(
                f"axis_gate_radius ({self._axis_gate_radius:+.4f}) 必须 > "
                f"preinsert_success_pos_threshold ({preinsert_success_pos_threshold:+.4f}); "
                "否则 ramp 区间退化, 门控逻辑无意义."
            )
        # Sphere-proxy 兜底参数. clearance_hard 允许 -inf (=关闭); 半径必须有限正数.
        self._clearance_hard = float(clearance_hard)
        self._proxy_arm_radius = float(proxy_arm_radius)
        self._proxy_ee_radius = float(proxy_ee_radius)
        self._exclude_ee_from_physx_self_collision = bool(
            exclude_ee_from_physx_self_collision
        )
        # Geometric preinsert 设置必须先于 obs dim 决定:
        # geom_stage 启用时隐含 use_axis_resid_obs=True, obs = AXIS_RESID 34 + 7 维几何 = 41 维.
        if geom_stage is None or geom_stage == "":
            self._geom_stage = None
        else:
            self._geom_stage = str(geom_stage).lower()
            if self._geom_stage not in ("prepos", "preaxis", "insert"):
                raise ValueError(
                    "geom_stage 必须是 None / 'prepos' / 'preaxis' / 'insert', "
                    f"got {geom_stage!r}"
                )
        # geom_stage 设计原则: dense reward + hold-N 仅作 eval 指标. 若 terminal_hold_bonus>0,
        # is_absorbing 会把 hold-N absorbing 接回 geom_success_mask, 等价于把 geom success
        # 变成 hard cliff (违反 Rule 1: 边界 Q-cliff; 也违反 cliff reward 训中心不训 floor).
        # 这是个容易误用的脚枪, 直接 raise 而不是 warn.
        if self._geom_stage is not None and self._terminal_hold_bonus > 0.0:
            raise ValueError(
                f"geom_stage={self._geom_stage!r} 与 terminal_hold_bonus={self._terminal_hold_bonus} > 0 互斥: "
                "几何路径必须靠 dense reward, 不用 hold-N absorbing 把 success 变 hard cliff. "
                "请关掉 terminal_hold_bonus (默认 0), 或切回 geom_stage=None."
            )
        if self._geom_stage is not None:
            use_axis_resid_obs = True

        self._use_axis_resid_obs = bool(use_axis_resid_obs)
        if self._geom_stage is not None:
            self._agent_obs_dim = AGENT_OBS_DIM_GEOM
        elif self._use_axis_resid_obs:
            self._agent_obs_dim = AGENT_OBS_DIM_AXIS_RESID
        else:
            self._agent_obs_dim = AGENT_OBS_DIM_BASE

        # Peg-shaft sample offsets — used by geom_stage radial_max / penetration_max.
        if peg_sample_offsets is None:
            offsets = DEFAULT_PEG_SAMPLE_OFFSETS
        else:
            offsets = tuple(float(x) for x in peg_sample_offsets)
            if any(o > 0.0 or o < -_PEG_HEIGHT for o in offsets):
                raise ValueError(
                    f"peg_sample_offsets 元素必须 ∈ [-{_PEG_HEIGHT:.3f}, 0]; "
                    f"got {offsets}"
                )
        self._peg_sample_offsets = offsets

        # Geometric preinsert (Stage 1g/2g/3g) presets. CLI 可逐项覆盖; 未覆盖时
        # 根据 geom_stage 选择安全默认值. 旧路径 geom_stage=None 时这些值闲置.
        geom_presets = {
            # progress=0 默认 → 跟旧 additive 行为完全等价. CLI 显式 >0 才开 gated.
            "prepos": dict(d=8.0, radial_tip=8.0, radial_max=0.0, axis=0.0, progress=0.0),
            "preaxis": dict(d=8.0, radial_tip=2.0, radial_max=5.0, axis=1.0, progress=0.0),
            "insert": dict(d=5.0, radial_tip=0.0, radial_max=5.0, axis=1.0, progress=0.0),
            None: dict(d=8.0, radial_tip=8.0, radial_max=0.0, axis=0.0, progress=0.0),
        }
        gp = geom_presets[self._geom_stage]
        self._w_geom_d = float(gp["d"] if rew_geom_d is None else rew_geom_d)
        self._w_geom_radial_tip = float(
            gp["radial_tip"] if rew_geom_radial_tip is None else rew_geom_radial_tip
        )
        self._w_geom_radial_max = float(
            gp["radial_max"] if rew_geom_radial_max is None else rew_geom_radial_max
        )
        self._w_geom_axis = float(gp["axis"] if rew_geom_axis is None else rew_geom_axis)
        self._w_geom_progress = float(
            gp["progress"] if rew_geom_progress is None else rew_geom_progress
        )
        self._geom_gate_radial_sigma = float(geom_gate_radial_sigma)
        self._geom_gate_axis_sigma = float(geom_gate_axis_sigma)
        self._w_geom_penetration = float(rew_geom_penetration or 0.0)
        # geom_gate_penetration_sigma=None → 不进 gate; finite>0 → 进 gate
        self._geom_gate_penetration_sigma = (
            float(geom_gate_penetration_sigma)
            if geom_gate_penetration_sigma is not None
            else None
        )
        if (self._geom_gate_penetration_sigma is not None
                and not (math.isfinite(self._geom_gate_penetration_sigma)
                         and self._geom_gate_penetration_sigma > 0.0)):
            raise ValueError(
                "geom_gate_penetration_sigma 必须 None (不用) 或 finite>0, "
                f"got {geom_gate_penetration_sigma}"
            )
        if not (math.isfinite(self._w_geom_penetration)
                and self._w_geom_penetration >= 0.0):
            raise ValueError(
                f"rew_geom_penetration must be finite and >= 0, got {self._w_geom_penetration}"
            )
        if cost_signal not in ("collision", "penetration"):
            raise ValueError(
                f"cost_signal 必须 'collision' 或 'penetration', got {cost_signal!r}"
            )
        self._cost_signal = cost_signal
        # d_target 必须先赋值, 因为下面 progress_floor 校验依赖它.
        self._geom_d_target_neg = float(geom_d_target_neg)
        self._geom_d_target_pos = float(geom_d_target_pos)
        self._geom_d_target_eff = self._geom_d_target_neg
        self._geom_progress_floor = float(geom_progress_floor)
        if not math.isfinite(self._geom_progress_floor):
            raise ValueError(
                f"geom_progress_floor must be finite, got {geom_progress_floor}"
            )
        # progress_floor 应该 ≥ d_target_neg (起点之前的位置), ≤ d_target_pos (终点),
        # 否则 progress 区间没意义.
        if not (self._geom_d_target_neg <= self._geom_progress_floor <= self._geom_d_target_pos):
            raise ValueError(
                f"geom_progress_floor ({self._geom_progress_floor:+.4f}) 必须 ∈ "
                f"[{self._geom_d_target_neg:+.4f}, {self._geom_d_target_pos:+.4f}] "
                "(d_target_neg, d_target_pos 区间内)"
            )
        # delta-progress (PBRS): phi 用 clean_gate × normalized_position
        self._w_geom_advance = float(rew_geom_advance or 0.0)
        if not (math.isfinite(self._w_geom_advance) and self._w_geom_advance >= 0.0):
            raise ValueError(
                f"rew_geom_advance must be finite and >= 0, got {self._w_geom_advance}"
            )
        # Task-ordering 修复 (codex 2026-05-11 v6)
        if geom_d_gate_mode not in ("off", "alignment"):
            raise ValueError(
                f"geom_d_gate_mode 必须 'off' 或 'alignment', got {geom_d_gate_mode!r}"
            )
        self._geom_d_gate_mode = geom_d_gate_mode
        self._w_geom_bad_entry = float(rew_geom_bad_entry or 0.0)
        if not (math.isfinite(self._w_geom_bad_entry) and self._w_geom_bad_entry >= 0.0):
            raise ValueError(
                f"rew_geom_bad_entry must be finite and >= 0, got {self._w_geom_bad_entry}"
            )
        self._geom_bad_entry_radial_safe = float(geom_bad_entry_radial_safe)
        self._geom_bad_entry_axis_safe = float(geom_bad_entry_axis_safe)
        self._geom_bad_entry_pen_safe = float(geom_bad_entry_pen_safe)
        for name, value in (
            ("geom_bad_entry_radial_safe", self._geom_bad_entry_radial_safe),
            ("geom_bad_entry_axis_safe", self._geom_bad_entry_axis_safe),
            ("geom_bad_entry_pen_safe", self._geom_bad_entry_pen_safe),
        ):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite > 0, got {value}")
        self._geom_d_target_ramp_start = int(geom_d_target_ramp_start)
        self._geom_d_target_ramp_end = int(geom_d_target_ramp_end)
        if self._geom_d_target_ramp_end < self._geom_d_target_ramp_start:
            raise ValueError(
                f"geom_d_target_ramp_end ({self._geom_d_target_ramp_end}) must >= "
                f"geom_d_target_ramp_start ({self._geom_d_target_ramp_start})"
            )
        self._geom_d_sat = float(geom_d_sat)
        self._geom_radial_sat = float(geom_radial_sat)
        self._w_geom_soft_success = float(rew_geom_soft_success)
        self._geom_soft_d_sigma = float(geom_soft_d_sigma)
        self._geom_soft_radial_sigma = float(geom_soft_radial_sigma)
        self._geom_soft_axis_sigma = float(geom_soft_axis_sigma)
        self._geom_soft_penetration_sigma = (
            float(geom_soft_penetration_sigma)
            if geom_soft_penetration_sigma is not None
            else None
        )
        if (self._geom_soft_penetration_sigma is not None
                and not (math.isfinite(self._geom_soft_penetration_sigma)
                         and self._geom_soft_penetration_sigma > 0.0)):
            raise ValueError(
                "geom_soft_penetration_sigma 必须 None (不用) 或 finite>0, "
                f"got {geom_soft_penetration_sigma}"
            )
        self._geom_d_th = float(geom_d_th)
        self._geom_r_tip_th = float(geom_r_tip_th)
        self._geom_r_max_th = float(geom_r_max_th)
        self._geom_axis_th = float(geom_axis_th)
        self._geom_insert_d_ins = float(geom_insert_d_ins)
        self._geom_insert_r_max_th = float(geom_insert_r_max_th)
        self._geom_pen_th = float(geom_pen_th)
        for name, value in (
            ("geom_d_sat", self._geom_d_sat),
            ("geom_radial_sat", self._geom_radial_sat),
            ("geom_soft_d_sigma", self._geom_soft_d_sigma),
            ("geom_soft_radial_sigma", self._geom_soft_radial_sigma),
            ("geom_soft_axis_sigma", self._geom_soft_axis_sigma),
            ("geom_d_th", self._geom_d_th),
            ("geom_r_tip_th", self._geom_r_tip_th),
            ("geom_r_max_th", self._geom_r_max_th),
            ("geom_axis_th", self._geom_axis_th),
            ("geom_insert_d_ins", self._geom_insert_d_ins),
            ("geom_insert_r_max_th", self._geom_insert_r_max_th),
            ("geom_pen_th", self._geom_pen_th),
            ("geom_gate_radial_sigma", self._geom_gate_radial_sigma),
            ("geom_gate_axis_sigma", self._geom_gate_axis_sigma),
        ):
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and > 0, got {value}")
        if not (math.isfinite(self._w_geom_progress) and self._w_geom_progress >= 0.0):
            raise ValueError(
                f"rew_geom_progress must be finite and >= 0, got {self._w_geom_progress}"
            )
        # gated progress 需要 d_target_pos > d_target_neg 才有意义的 progress 区间.
        # 只在 insert 模式或显式开启 gated progress (w_progress > 0) 时强制此约束,
        # 防止 prepos / preaxis 调试时误伤 (那两个 stage 不读 d_target_pos).
        _gated_active = (
            self._geom_stage == "insert" or self._w_geom_progress > 0.0
        )
        if _gated_active and self._geom_d_target_pos <= self._geom_d_target_neg:
            raise ValueError(
                f"geom_d_target_pos ({self._geom_d_target_pos:+.4f}) must > "
                f"geom_d_target_neg ({self._geom_d_target_neg:+.4f}) "
                f"when geom_stage=insert or rew_geom_progress > 0; "
                "gated progress range otherwise 是负的或 0, reward 信号无效."
            )
        # geom_stage 启用时: obs 里 pos_vec = peg_tip - (hole_entry + preinsert_offset·hole_axis),
        # reward 里 d_target_eff 从 geom_d_target_neg 起算. 若 preinsert_offset != abs(d_target_neg),
        # obs target 和 reward target 是不同深度, agent 学到的 anchor 跟 reward 实际推的目标错位.
        # train_sac/eval/viz 已在 CLI 端自动同步, 但直接构造 env (e.g. notebook / 调试脚本) 时
        # 没人兜底, 所以在 env 端 fail-fast.
        if self._geom_stage is not None:
            expected_offset = abs(self._geom_d_target_neg)
            if abs(self._preinsert_offset - expected_offset) > 1e-4:
                raise ValueError(
                    f"geom_stage={self._geom_stage!r} 启用时, preinsert_offset "
                    f"(={self._preinsert_offset:+.4f}) 必须等于 abs(geom_d_target_neg) "
                    f"(={expected_offset:.4f}); 否则 obs 里 pos_vec 的 target 跟 reward 的 "
                    "d_target 是两个不同深度, agent 学到的 anchor 与 reward 实际推的目标错位."
                )
        if not (math.isfinite(self._proxy_arm_radius) and self._proxy_arm_radius > 0.0):
            raise ValueError(
                f"proxy_arm_radius 必须是有限正数, 传入 {proxy_arm_radius}"
            )
        if not (math.isfinite(self._proxy_ee_radius) and self._proxy_ee_radius > 0.0):
            raise ValueError(
                f"proxy_ee_radius 必须是有限正数, 传入 {proxy_ee_radius}"
            )
        self._usd_path = Path(usd_path) if usd_path is not None else DEFAULT_USD_PATH
        if not self._usd_path.is_file():
            raise FileNotFoundError(
                "找不到机器人 USD 资产文件: "
                f"{self._usd_path}\n"
                "请确认仓库内存在 assets/usd/dual_arm_iiwa/dual_arm_iiwa_with_peghole.usda，"
                "或在构造 DualArmPegHoleEnv 时显式传入 usd_path。"
            )

        # is_absorbing 与 reward 在同一 next_obs 上背靠背调用, 缓存避免重复计算.
        # _create_observation 每步会刷新这些 cache.
        self._last_collision_mask = None
        self._last_pos_err = None
        self._last_axis_err = None
        self._last_success_mask = None
        self._last_pos_success_mask = None
        # sphere-proxy clearance: is_absorbing 里每步算并 cache,
        # _last_min_clearance < clearance_hard 即触发 hard absorbing.
        self._last_min_clearance = None
        # _preprocess_action → reward 链路里缓存 pre-scale 的 raw action,
        # 用于 L2 惩罚. 这样 w_action 和 action_scale 解耦.
        self._last_raw_action = None
        # Geom 几何 cache — 仅 geom_stage 非空时由 _create_observation 写入,
        # _compute_geom_reward_components 读. Lagrangian (None) 路径不读这些字段.
        self._cached_d = None
        self._cached_radial_vec = None
        self._cached_radial_err_tip = None
        self._cached_radial_max = None
        self._cached_penetration_max = None
        self._cached_peg_in_hole = None
        # phi (PBRS potential) per-env cache. None = 还没构造; 构造后是 tensor (n_envs,).
        # _phi_first_step_mask[i]=True 表示 env i 刚 reset, 第一次 reward 计算时
        # Δphi 应当置 0 (上一 episode 末态 phi 不能跨 episode 用做 prev).
        self._cached_phi_prev = None
        self._phi_first_step_mask = None

        observation_spec = [
            ("joint_pos", "", ObservationType.JOINT_POS, ARM_JOINTS),
            ("joint_vel", "", ObservationType.JOINT_VEL, ARM_JOINTS),
            ("left_ee_pos", LEFT_EE_PATH, ObservationType.BODY_POS, None),
            ("right_ee_pos", RIGHT_EE_PATH, ObservationType.BODY_POS, None),
            ("left_ee_rot", LEFT_EE_PATH, ObservationType.BODY_ROT, None),
            ("right_ee_rot", RIGHT_EE_PATH, ObservationType.BODY_ROT, None),
        ]
        if self._exclude_ee_from_physx_self_collision:
            collision_groups = [("arm_L", LEFT_ARM_LINKS), ("arm_R", RIGHT_ARM_LINKS)]
        else:
            collision_groups = [("arm_L", LEFT_ARM_GROUP), ("arm_R", RIGHT_ARM_GROUP)]

        super().__init__(
            usd_path=str(self._usd_path),
            actuation_spec=ARM_JOINTS,
            observation_spec=observation_spec,
            backend="torch",
            device=device,
            collision_between_envs=False,
            num_envs=num_envs,
            env_spacing=4.0,
            gamma=gamma,
            horizon=horizon,
            timestep=0.02,
            n_intermediate_steps=5,
            action_type=ActionType.VELOCITY,
            collision_groups=collision_groups,
            headless=headless,
            camera_position=(20, -15, 10),
            camera_target=(5, 0, 0.5),
        )

        # 关节位限 + 默认位
        limits = self._task.get_joint_pos_limits()
        self._joint_lower, self._joint_upper = limits[0], limits[1]
        # 不用 USD 自带的 zero pose, 改用 HOME_JOINT_POS (胸前 ready), 避免 reset
        # center 落在 zero 全展开姿态, Stage 1 早期探索浪费在无效扇区.
        self._default_joint_pos = torch.tensor(
            HOME_JOINT_POS, device=device, dtype=self._joint_lower.dtype
        )
        self._home_weights = torch.tensor(
            self._home_weights_values, device=device, dtype=self._joint_lower.dtype
        )
        # fail-fast: home pose 必须落在每个关节的 [lower, upper] 内, 不然 reset
        # 那一步 PhysX 会把它 clamp 到边界, 与设计意图不符.
        if torch.any(self._default_joint_pos < self._joint_lower) or torch.any(
            self._default_joint_pos > self._joint_upper
        ):
            bad = (
                (self._default_joint_pos < self._joint_lower)
                | (self._default_joint_pos > self._joint_upper)
            ).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(
                f"HOME_JOINT_POS 越界: 关节索引 {bad} 不在 [lower, upper] 内. "
                f"lower={self._joint_lower.tolist()}  upper={self._joint_upper.tolist()}  "
                f"home={self._default_joint_pos.tolist()}"
            )

        # USD iiwa 默认 position-drive (kp ~ 5e5); velocity 控制必须把 kp 置 0,
        # 否则 reset 里 set_joint_positions 会一并写 pos_target, 高 kp 把关节钉
        # 回 reset 点.
        robots = self._task.robots
        cj = self._task._controlled_joints
        zero_kps = torch.zeros(self._n_envs, len(ARM_JOINTS), device=device)
        _, cur_kds = robots.get_gains(joint_indices=cj, clone=True)
        robots.set_gains(kps=zero_kps, kds=cur_kds, joint_indices=cj)
        self._cj = cj
        self._robots = robots

        # 累计自碰撞终止次数 (train_sac 每 epoch 读取并差分).
        # _absorb_count       — 总数 (任一信号触发, 反映 episode 终止次数)
        # _absorb_count_physx — PhysX 力检测触发数
        # _absorb_count_sphere— sphere-proxy clearance 触发数
        # PhysX 与 sphere 可同步触发, 所以 physx + sphere ≥ total.
        self._absorb_count = 0
        self._absorb_count_physx = 0
        self._absorb_count_sphere = 0

        # hold-N 计数器 (per env)
        self._consecutive_inthresh = torch.zeros(
            self._n_envs, dtype=torch.long, device=self._device
        )
        self._last_hold_done_mask = None

        # 解析式 frame 用的常量 (LeftEE 局部坐标), 一次 build, broadcast 用
        dtype = self._joint_lower.dtype
        dev = self._device
        self._peg_tip_offset = torch.tensor(PEG_TIP_OFFSET_IN_LEFTEE, device=dev, dtype=dtype)
        self._hole_entry_offset = torch.tensor(HOLE_ENTRY_OFFSET_IN_RIGHTEE, device=dev, dtype=dtype)
        self._peg_axis_local = torch.tensor(PEG_AXIS_IN_LEFTEE, device=dev, dtype=dtype)
        self._hole_axis_local = torch.tensor(HOLE_AXIS_IN_RIGHTEE, device=dev, dtype=dtype)
        self._peg_axis_quat_offset = torch.tensor(
            PEG_AXIS_QUAT_OFFSET, device=dev, dtype=dtype
        )
        self._hole_axis_quat_offset = torch.tensor(
            HOLE_AXIS_QUAT_OFFSET, device=dev, dtype=dtype
        )

        # peg/hole 资产存在性 fail-fast 检查 (phase 1.5 commit 2 的 print 降级删除)
        self._verify_peghole_prims_exist()

        # Sphere-proxy 索引 + 半径 tensor 解析. 必须在 super().__init__() 之后,
        # body_names 才存在. is_absorbing 每步调 _compute_min_clearance().
        self._build_sphere_proxy_indices()

        # 同步一次物理状态, 避免 reset_all 后第一帧 BODY_POS / BODY_ROT 是 stale
        self._world.step(render=False)

    # ------------------------------------------------------------------
    # mushroom hooks
    # ------------------------------------------------------------------
    def _modify_mdp_info(self, mdp_info):
        # action: [-1,1]^14, SAC tanh policy 直接映射
        device = mdp_info.action_space.low.device
        dtype = mdp_info.action_space.low.dtype
        one = torch.ones(len(ARM_JOINTS), device=device, dtype=dtype)
        mdp_info.action_space = Box(-one, one, data_type=dtype)

        # observation: 32 维 agent obs (见模块 docstring + _AGENT_OBS_* 切片).
        # 不能用 self._joint_lower / _joint_upper 在这里取 — 它们在 super().__init__
        # 之后才赋值, 而 mushroom 在 super 里就调本函数. obs_helper 已经构造完毕,
        # 走它的 obs_limits + obs_idx_map 切出 joint 段.
        raw_low, raw_high = self.observation_helper.obs_limits
        jp_idx = self.observation_helper.obs_idx_map["joint_pos"]
        jv_idx = self.observation_helper.obs_idx_map["joint_vel"]
        jp_low = raw_low[jp_idx].to(dtype)
        jp_high = raw_high[jp_idx].to(dtype)
        jv_low = raw_low[jv_idx].to(dtype)
        jv_high = raw_high[jv_idx].to(dtype)
        pos_lo = torch.full((3,), -5.0, device=jp_low.device, dtype=dtype)
        pos_hi = torch.full((3,), 5.0, device=jp_low.device, dtype=dtype)
        if self._use_axis_resid_obs or self._geom_stage is not None:
            # axis_resid = peg_axis + hole_axis, 每分量 ∈ [-2, +2] (两个单位向量和).
            resid_lo = torch.full((3,), -2.0, device=jp_low.device, dtype=dtype)
            resid_hi = torch.full((3,), +2.0, device=jp_low.device, dtype=dtype)
            chunks_low = [jp_low, jv_low, pos_lo, resid_lo]
            chunks_high = [jp_high, jv_high, pos_hi, resid_hi]
        else:
            axis_lo = torch.full((1,), -1.0, device=jp_low.device, dtype=dtype)
            axis_hi = torch.full((1,), 1.0, device=jp_low.device, dtype=dtype)
            chunks_low = [jp_low, jv_low, pos_lo, axis_lo]
            chunks_high = [jp_high, jv_high, pos_hi, axis_hi]
        if self._geom_stage is not None:
            # Geometric preinsert 末尾 7 维:
            #   d (1D, 深度) + radial_vec (3D, world-frame 法向)
            #   + peg_in_hole (3D, world-frame 相对). 所有量都是米, 半世界尺度足够.
            d_lo = torch.full((1,), -0.5, device=jp_low.device, dtype=dtype)
            d_hi = torch.full((1,), +0.5, device=jp_low.device, dtype=dtype)
            radvec_lo = torch.full((3,), -0.5, device=jp_low.device, dtype=dtype)
            radvec_hi = torch.full((3,), +0.5, device=jp_low.device, dtype=dtype)
            peg_lo = torch.full((3,), -1.0, device=jp_low.device, dtype=dtype)
            peg_hi = torch.full((3,), +1.0, device=jp_low.device, dtype=dtype)
            chunks_low.extend([d_lo, radvec_lo, peg_lo])
            chunks_high.extend([d_hi, radvec_hi, peg_hi])
        new_obs_low = torch.cat(chunks_low, dim=0)
        new_obs_high = torch.cat(chunks_high, dim=0)
        mdp_info.observation_space = Box(new_obs_low, new_obs_high, data_type=dtype)
        return mdp_info

    def _preprocess_action(self, action):
        action = torch.as_tensor(action, device=self._device, dtype=self._joint_lower.dtype)
        clipped = torch.clamp(action, -1.0, 1.0)
        self._last_raw_action = clipped
        return clipped * self._action_scale

    def _create_observation(self, obs):
        """raw obs (42 dim) → agent obs (32 / 34 / 41 dim).

        raw 布局 (与 observation_spec 顺序一致):
            joint_pos[14] joint_vel[14] left_ee_pos[3] right_ee_pos[3]
            left_ee_rot[4] right_ee_rot[4]
        agent obs 布局 (见 _AGENT_OBS_* 切片):
            joint_pos[14] joint_vel[14] pos_vec[3] axis_dot[1]                       ← 32D
            joint_pos[14] joint_vel[14] pos_vec[3] axis_resid[3]                     ← 34D
            joint_pos[14] joint_vel[14] pos_vec[3] axis_resid[3]
                d[1] radial_vec[3] peg_in_hole[3]                                    ← 41D (geom)
        """
        joint_pos = self.observation_helper.get_from_obs(obs, "joint_pos")
        joint_vel = self.observation_helper.get_from_obs(obs, "joint_vel")
        left_ee = self.observation_helper.get_from_obs(obs, "left_ee_pos")
        right_ee = self.observation_helper.get_from_obs(obs, "right_ee_pos")
        left_quat = self.observation_helper.get_from_obs(obs, "left_ee_rot")
        right_quat = self.observation_helper.get_from_obs(obs, "right_ee_rot")

        peg_tip = left_ee + self._quat_apply(left_quat, self._peg_tip_offset)
        hole_entry = right_ee + self._quat_apply(right_quat, self._hole_entry_offset)
        peg_axis = self._quat_apply(left_quat, self._peg_axis_local)
        hole_axis = self._quat_apply(right_quat, self._hole_axis_local)
        peg_axis = peg_axis / peg_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        hole_axis = hole_axis / hole_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        preinsert_target = hole_entry + self._preinsert_offset * hole_axis
        pos_vec = peg_tip - preinsert_target

        # axis_dot ∈ [-1, +1], -1 = 完美对齐. axis_err = 1 + axis_dot ∈ [0, 2].
        # 三种 obs 模式都用同一个 axis_err 语义 (1+dot), 让 reward / eval / viz
        # / success_axis_threshold 全程一致, 避免 rad-vs-(1+dot) 量纲混淆.
        axis_dot = (peg_axis * hole_axis).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        axis_err = 1.0 + axis_dot.squeeze(-1)

        # cache 给 is_absorbing / reward 复用 (避免再算一遍 quat 旋转).
        self._cached_pos_vec = pos_vec
        self._cached_axis_err = axis_err

        if self._geom_stage is not None:
            # Geometric preinsert 几何 (世界帧表达, 没做 hole-local rotation):
            #   d = -axial_dist 让 d>0 = inserted.
            #   axial_dist = (peg_tip - hole_entry) · hole_axis. sign 验证见
            #   scripts/archive/sanity_eval_stage3.py (在 preinsert axial_dist ≈ +preinsert_offset).
            peg_in_hole = peg_tip - hole_entry
            axial_dist = (peg_in_hole * hole_axis).sum(dim=-1)
            d_depth = -axial_dist
            radial_vec = peg_in_hole - axial_dist.unsqueeze(-1) * hole_axis
            radial_err_tip = torch.norm(radial_vec, dim=-1)

            # radial_max: 沿 peg 反向多采样点, 取径向距离 max. 杆身斜了 tip 单点严重低估.
            # penetration_max (codex 2026-05-11 v2): peg 表面相对 hole 内壁的物理
            # 穿模量. 三个 fix vs v1:
            #   (a) 加 radial 区间检查: sample 必须 radial_s < OUTER+PEG_R, 否则
            #       远场 peg 即使 axial 落入范围也不算"in wall" (v1 给过 1316mm 假数据)
            #   (b) 加上限 clamp: penetration 物理上限 = wall_thickness = 4mm
            #       (peg 完全穿过 4mm wall 后, peg 已在外侧空间, 不再"in wall")
            #   (c) 完全 dense, 不依赖 PhysX contact, 不引入 absorb
            radial_max = radial_err_tip.clone()
            wall_intersect_radial_limit = _HOLE_OUTER_RADIUS + _PEG_RADIUS  # 0.040m
            # tip 样本 (offset=0) 也参与: 复用 axial_dist / radial_err_tip 节省计算.
            tip_in_wall = (
                (axial_dist < 0.0) & (axial_dist > -_HOLE_DEPTH)
                & (radial_err_tip < wall_intersect_radial_limit)
            )
            tip_penetration = torch.where(
                tip_in_wall,
                torch.clamp(
                    radial_err_tip + _PEG_RADIUS - _HOLE_INNER_RADIUS,
                    min=0.0, max=_HOLE_WALL_THICKNESS,
                ),
                torch.zeros_like(radial_err_tip),
            )
            penetration_max = tip_penetration
            for off in self._peg_sample_offsets:
                if off == 0.0:
                    continue
                sample_pos = peg_tip + off * peg_axis
                d_s = sample_pos - hole_entry
                axial_s = (d_s * hole_axis).sum(dim=-1)
                radial_v_s = d_s - axial_s.unsqueeze(-1) * hole_axis
                rad_s = torch.norm(radial_v_s, dim=-1)
                radial_max = torch.maximum(radial_max, rad_s)
                # 双重检查 + clamp 后 penetration ∈ [0, 0.004m]
                sample_in_wall = (
                    (axial_s < 0.0) & (axial_s > -_HOLE_DEPTH)
                    & (rad_s < wall_intersect_radial_limit)
                )
                pen_s = torch.where(
                    sample_in_wall,
                    torch.clamp(
                        rad_s + _PEG_RADIUS - _HOLE_INNER_RADIUS,
                        min=0.0, max=_HOLE_WALL_THICKNESS,
                    ),
                    torch.zeros_like(rad_s),
                )
                penetration_max = torch.maximum(penetration_max, pen_s)

            self._cached_d = d_depth
            self._cached_radial_vec = radial_vec
            self._cached_radial_err_tip = radial_err_tip
            self._cached_radial_max = radial_max
            self._cached_penetration_max = penetration_max
            self._cached_peg_in_hole = peg_in_hole

            axis_resid = peg_axis + hole_axis
            return torch.cat([
                joint_pos, joint_vel, pos_vec, axis_resid,
                d_depth.unsqueeze(-1), radial_vec, peg_in_hole,
            ], dim=-1)

        if self._use_axis_resid_obs:
            # axis_resid = peg_axis + hole_axis. 同向 (home pose) 模长 2,
            # 反向 (完美对齐) 模长 0. 信息上等价于 axis_dot 但 3D 带方向,
            # 全程光滑无奇异. ||resid||² = 2·(1 + dot) = 2·axis_err.
            axis_resid = peg_axis + hole_axis
            return torch.cat([joint_pos, joint_vel, pos_vec, axis_resid], dim=-1)
        return torch.cat([joint_pos, joint_vel, pos_vec, axis_dot], dim=-1)

    def _compute_task_errors(self, agent_obs):
        """从 agent obs 切片重建 (pos_err, axis_err, success_mask).

        train_sac.py 的 reset stats 和 _eval_utils.compute_hold_metrics 都通过
        这条接口拉指标; agent obs 已经包含 pos_vec 和 axis_dot, 所以不需要再走
        _create_observation 重新查 EE pose.

        success 用 stage flag 控制:
            success_axis_threshold=inf 时 axis 项恒 True, 退化为 pos-only (Stage 1 行为).
            success_axis_threshold 取有限值时变成 pos ∧ axis (Stage 2 训练用 0.50,
            严格 eval 用 0.30 / 0.20).
        """
        pos_vec = agent_obs[..., _AGENT_OBS_POS_VEC]
        pos_err = torch.norm(pos_vec, dim=-1)
        if self._use_axis_resid_obs:
            # axis_resid = peg + hole, ||resid||² = 2·(1 + dot) = 2·axis_err.
            # 反算 axis_err 与 reward / _cached_axis_err 同语义 (1 + dot).
            resid = agent_obs[..., _AGENT_OBS_AXIS_RESID]
            axis_err = (resid * resid).sum(dim=-1) / 2.0
        else:
            # axis_dot 已经在 obs 里 [-1, +1]; axis_err = 1 + axis_dot ∈ [0, 2].
            axis_dot = agent_obs[..., _AGENT_OBS_AXIS_DOT].squeeze(-1)
            axis_err = 1.0 + axis_dot
        success_mask = (
            (pos_err < self._preinsert_success_pos_threshold)
            & (axis_err < self._success_axis_threshold)
        )
        return pos_err, axis_err, success_mask

    def _compute_geom_success_masks(self):
        """Geometric preinsert success masks from cached 41D geom geometry.

        prepos:  axial depth target + tip radial.
        preaxis: prepos plus whole-peg radial_max + dense axis threshold.
        insert:  positive insertion depth + radial_max + axis threshold.

        These masks are for eval / best_hold / optional hold-N only. They are not
        used as hard reward cliffs in _compute_geom_reward_components.
        """
        d = self._cached_d
        radial_tip = self._cached_radial_err_tip
        radial_max = self._cached_radial_max
        penetration_max = self._cached_penetration_max
        axis_err = self._cached_axis_err
        d_err = torch.abs(d - self._geom_d_target_eff)
        prepos_mask = (
            (d_err < self._geom_d_th)
            & (radial_tip < self._geom_r_tip_th)
        )
        preaxis_mask = (
            prepos_mask
            & (radial_max < self._geom_r_max_th)
            & (axis_err < self._geom_axis_th)
        )
        # insert_mask 加 `d_err < geom_d_th` (codex 2026-05-11 fix): 防止 peg
        # overshoot (d=+0.08, target=+0.03) 被算成功. 没这个深度窗口, mask 只要
        # peg 越过 d_ins=0.025 就 True, 即使 overshoot 5cm. 加了后 success window
        # 是 d ∈ [d_ins, d_target_eff + d_th] (典型 [0.025, 0.050]).
        # 2026-05-11 v4: 同时要求 penetration_max < geom_pen_th, 否则
        # best_hold 会保存 "mask 满足但穿过 hole wall" 的假成功状态.
        insert_mask = (
            (d > self._geom_insert_d_ins)
            & (d_err < self._geom_d_th)
            & (radial_max < self._geom_insert_r_max_th)
            & (axis_err < self._geom_axis_th)
            & (penetration_max < self._geom_pen_th)
        )
        if self._geom_stage == "prepos":
            active_mask = prepos_mask
        elif self._geom_stage == "preaxis":
            active_mask = preaxis_mask
        elif self._geom_stage == "insert":
            active_mask = insert_mask
        else:
            active_mask = torch.zeros_like(prepos_mask)
        return prepos_mask, preaxis_mask, insert_mask, active_mask

    def is_absorbing(self, obs):
        physx_collision = self._check_collision("arm_L", "arm_R", self._collision_threshold,
                                                dt=self._timestep)
        # sphere-proxy 兜底: 双臂 sphere proxy 的最小 clearance 跌破 clearance_hard
        # 也算 collision. clearance_hard=-inf 时此项恒 False, 退化为纯 PhysX.
        min_clearance, _ = self._compute_min_clearance()
        self._last_min_clearance = min_clearance
        if math.isfinite(self._clearance_hard):
            sphere_collision = min_clearance < self._clearance_hard
        else:
            sphere_collision = torch.zeros_like(physx_collision)
        collision = physx_collision | sphere_collision
        # 两个 bucket 可同时触发 (一步同时撞), 分别累加便于诊断哪个信号在主导;
        # _absorb_count 仍按 OR 后的 collision 累加 (= 实际 absorb 次数).
        self._absorb_count_physx += int(physx_collision.sum().item())
        self._absorb_count_sphere += int(sphere_collision.sum().item())
        self._absorb_count += int(collision.sum().item())
        self._last_collision_mask = collision

        # _create_observation 已 cache pos_vec / axis_err; 在这里只 compose success.
        # axis_th=inf (Stage 1) 时 axis 项恒 True, success 退化为 pos-only.
        pos_err = torch.norm(self._cached_pos_vec, dim=-1)
        axis_err = self._cached_axis_err
        pos_in_thresh = pos_err < self._preinsert_success_pos_threshold
        if self._geom_stage is not None:
            geom_prepos, _, _, geom_success = self._compute_geom_success_masks()
            success_mask = geom_success
            pos_in_thresh = geom_prepos
        else:
            success_mask = pos_in_thresh & (axis_err < self._success_axis_threshold)
        self._last_pos_err = pos_err
        self._last_axis_err = axis_err
        self._last_pos_success_mask = pos_in_thresh
        self._last_success_mask = success_mask

        # 更新 per-env 的连续 in-threshold 计数
        self._consecutive_inthresh = torch.where(
            success_mask,
            self._consecutive_inthresh + 1,
            torch.zeros_like(self._consecutive_inthresh),
        )

        if self._absorbing_terminal_active:
            hold_done = self._consecutive_inthresh >= self._success_hold_steps
        else:
            hold_done = torch.zeros_like(collision)
        self._last_hold_done_mask = hold_done

        return collision | hold_done

    def _create_info_dictionary(self, obs):
        # cost = constraint cost signal for Lagrangian SAC.
        # cost_signal='collision' (default, 老语义): 0/1 indicator from PhysX OR
        #   sphere-proxy. is_absorbing 已 cache _last_collision_mask.
        # cost_signal='penetration': 连续 [0, 4mm] = peg 表面相对 hole 内壁的
        #   physical overlap. 几何信号, 不依赖 PhysX contact. 适合 Lagrangian SAC.
        if self._cost_signal == "penetration" and self._cached_penetration_max is not None:
            cost = self._cached_penetration_max.to(torch.float32)
        elif self._last_collision_mask is None:
            cost = torch.zeros(self._n_envs, dtype=torch.float32, device=self._device)
        else:
            cost = self._last_collision_mask.to(torch.float32)
        info = {"cost": cost}
        if self._geom_stage is not None and self._cached_d is not None:
            geom_prepos, geom_preaxis, geom_insert, geom_success = (
                self._compute_geom_success_masks()
            )
            info["geom_d"] = self._cached_d.to(torch.float32)
            info["geom_d_target"] = torch.full_like(
                self._cached_d, self._geom_d_target_eff, dtype=torch.float32
            )
            info["geom_radial_tip"] = self._cached_radial_err_tip.to(torch.float32)
            info["geom_radial_max"] = self._cached_radial_max.to(torch.float32)
            info["geom_axis_err"] = self._cached_axis_err.to(torch.float32)
            # penetration_max: peg 表面相对 hole 内壁的几何穿模量, > 0 = 物理穿模.
            # 不进 reward / mask (step 1 instrumentation only), 仅写 info 给 eval
            # 聚合穿模分布. 后续 step 2/3 决定是否进 reward.
            info["geom_penetration_max"] = self._cached_penetration_max.to(torch.float32)
            info["geom_prepos_mask"] = geom_prepos.to(torch.float32)
            info["geom_preaxis_mask"] = geom_preaxis.to(torch.float32)
            info["geom_insert_mask"] = geom_insert.to(torch.float32)
            info["geom_success_mask"] = geom_success.to(torch.float32)
        return info

    def _compute_joint_limit_norm(self, joint_pos):
        """归一化关节越限惩罚: 各 DoF 越 margin 后线性 ramp [margin_frac, 1] → [0, 1],
        平方求和. 单位无量纲, w_joint_limit 在 Stage 1/2/3 之间量纲一致."""
        joint_range = self._joint_upper - self._joint_lower
        joint_center = 0.5 * (self._joint_upper + self._joint_lower)
        excess = torch.clamp(
            (torch.abs((joint_pos - joint_center) / (0.5 * joint_range))
             - self._joint_limit_margin_frac)
            / (1.0 - self._joint_limit_margin_frac),
            min=0.0, max=1.0,
        )
        return torch.sum(excess ** 2, dim=-1)

    def _compute_axis_gate(self, pos_err):
        """axis 惩罚的距离门控: pos_err >= gate_radius 时关 (gate=0); 进入
        [pos_th, gate_radius] 区间线性 ramp; pos 进阈后 gate=1.
        gate_radius=inf 退化为不门控."""
        if math.isfinite(self._axis_gate_radius):
            denom = max(self._axis_gate_radius - self._preinsert_success_pos_threshold, 1e-6)
            return ((self._axis_gate_radius - pos_err) / denom).clamp(0.0, 1.0)
        return torch.ones_like(pos_err)

    def _compute_home_norm(self, joint_pos):
        joint_range = self._joint_upper - self._joint_lower
        home_dev = (joint_pos - self._default_joint_pos.unsqueeze(0)) / joint_range.unsqueeze(0)
        return (self._home_weights.unsqueeze(0) * (home_dev ** 2)).sum(dim=-1)

    def _compute_normal_reward(self, next_obs):
        """[SUPERSEDED 2026-05-12 — kept as baseline; main line is `_compute_geom_reward_components` via `--geom_stage`.]

        旧 Stage 1/2 球形 pos_err shaped reward, 不含 collision/hold-N absorbing 分支.
        DualArmPegHoleCostEnv (Lagrangian 路径) 仍调用. 不要扩展.
        """
        joint_pos = next_obs[..., _AGENT_OBS_JOINT_POS]
        pos_err = self._last_pos_err
        axis_err = self._last_axis_err
        full_success = self._last_success_mask.to(pos_err.dtype)
        pos_success = self._last_pos_success_mask.to(pos_err.dtype)

        joint_limit_norm = self._compute_joint_limit_norm(joint_pos)
        action_sq = (self._last_raw_action ** 2).sum(dim=-1)
        home_norm = self._compute_home_norm(joint_pos)
        axis_gate = self._compute_axis_gate(pos_err)

        return (
            -self._w_pos * pos_err
            - self._w_axis * axis_gate * axis_err
            - self._w_joint_limit * joint_limit_norm
            - self._w_action * action_sq
            - self._w_home * home_norm
            + self._w_pos_success * pos_success
            + self._w_success * full_success
        )

    def _compute_geom_reward_components(self, next_obs):
        """Geometric preinsert reward components.

        This path is the root-cause replacement for old spherical pos_err
        preinsert rewards. It uses the same 41D geometry as Stage 3, but keeps
        the old learning order:

        - prepos:  depth target + tip radial only (no hidden axis pressure).
        - preaxis: add radial_max and axis dense terms.
        - insert:  same geometry, with d_target scheduled toward positive depth.

        Success masks are deliberately excluded from the reward by default. The
        optional soft well is Gaussian and off unless rew_geom_soft_success > 0.

        Optional alignment-gated progress (codex 提案 2026-05-11): when
        rew_geom_progress > 0, adds
            +w_progress · clamp(d - d_target_neg, 0, d_target_pos - d_target_neg)
                        · exp(-(rm/σ_r)² - (axis/σ_a)²)
        让"轴向推进"reward 只在 radial/axis 同时好时才被取到. 解决 additive
        reward 不强制 joint 满足的问题. additive 项 (r_geom_d / radial / axis)
        仍保留, 用户可通过 CLI 把 r_geom_d 设 0 让 gated progress 主导, 或保留
        作 anchor.
        """
        joint_pos = next_obs[..., _AGENT_OBS_JOINT_POS]
        d = self._cached_d
        radial_tip = self._cached_radial_err_tip
        radial_max = self._cached_radial_max
        axis_err = self._cached_axis_err

        d_err = torch.abs(d - self._geom_d_target_eff)
        d_err_sat = torch.clamp(d_err, max=self._geom_d_sat)
        radial_tip_sat = torch.clamp(radial_tip, max=self._geom_radial_sat)
        radial_max_sat = torch.clamp(radial_max, max=self._geom_radial_sat)
        penetration_max = self._cached_penetration_max

        # alignment_gate 提到前面 (codex 2026-05-11 v6), 因为 r_d 现在可选乘它
        # 来强制 task ordering. 公式跟下面 progress / advance 用的同一份.
        gate_exponent = (
            - (radial_max / self._geom_gate_radial_sigma) ** 2
            - (axis_err / self._geom_gate_axis_sigma) ** 2
        )
        if self._geom_gate_penetration_sigma is not None:
            gate_exponent = gate_exponent - (
                penetration_max / self._geom_gate_penetration_sigma
            ) ** 2
        alignment_gate = torch.exp(gate_exponent)

        # r_geom_d: 可选 gating (codex 2026-05-11 v6).
        # mode="off" (默认): r_d = -w · d_err_sat, 跟旧行为完全一致
        # mode="alignment": r_d = -w · d_err_sat · alignment_gate
        # ⚠️ SIGN WARNING (codex 2026-05-11 v6 catch): r_d 是 negative penalty.
        # 乘 alignment_gate 后, "misaligned → penalty 消失" 反而**激励 misalignment**
        # (aligned 时暴露 d_err penalty, misaligned 时藏起来). 用 mode="alignment"
        # 当心这个 perverse incentive. 推荐做法是 --rew_geom_d 0 关掉 d anchor,
        # 让 r_advance (positive, gate 正乘 → 同号) 主管 depth, 不用 r_d gating.
        r_geom_d_raw = -self._w_geom_d * d_err_sat
        if self._geom_d_gate_mode == "alignment":
            r_geom_d = r_geom_d_raw * alignment_gate
        else:
            r_geom_d = r_geom_d_raw
        r_geom_radial_tip = -self._w_geom_radial_tip * radial_tip_sat
        r_geom_radial_max = -self._w_geom_radial_max * radial_max_sat
        r_geom_axis = -self._w_geom_axis * axis_err

        # r_geom_bad_entry (codex 2026-05-11 v6.1 normalized): 显式 task ordering penalty.
        # 公式 (normalized): -w × depth_norm × Σ(relu(metric/safe - 1))
        #   depth_norm = clamp(d / d_target_pos, 0, 2)     ← 越过 entrance 才生效
        #   *_violation = relu(metric/safe - 1)             ← 单位 = "几倍 safe 阈值"
        #
        # **v6.0 → v6.1 修复 scale bug**: 旧公式 -w × d × (metric - safe) 中,
        # d 是米 (0.03), violations 是米 (0.001~0.01), 乘积 ~1e-4. 即使 w=50
        # 实际只罚 -0.004/step, 被 dwell +1/step 完全淹没. 归一化后 w=1.0 量级合理:
        # 1cm 超 1cm 阈值 (radial 大 100%) + d 在 target 处 → -1.0/step.
        # w_bad_entry=0 (默认) 关闭, 跟旧行为完全一致.
        if self._w_geom_bad_entry > 0.0:
            d_pos_safe = max(1e-6, self._geom_d_target_pos)
            depth_norm = torch.clamp(d / d_pos_safe, min=0.0, max=2.0)
            # codex 2026-05-12 v6.2 fix: per-violation clamp at 3.0 + total clamp at 3.0.
            # 没 cap 时 radial=1.4m → radial_v=139 → penalty -139/step 打爆 critic
            # (v7a ep 7 J=-4598 实证). cap 后 max penalty = w·2·3 = 6w/step, 强但不毁.
            radial_violation = torch.clamp(
                radial_max / self._geom_bad_entry_radial_safe - 1.0,
                min=0.0, max=3.0,
            )
            axis_violation = torch.clamp(
                axis_err / self._geom_bad_entry_axis_safe - 1.0,
                min=0.0, max=3.0,
            )
            pen_violation = torch.clamp(
                penetration_max / self._geom_bad_entry_pen_safe - 1.0,
                min=0.0, max=3.0,
            )
            total_violation = torch.clamp(
                radial_violation + axis_violation + pen_violation,
                max=3.0,
            )
            r_geom_bad_entry = -self._w_geom_bad_entry * depth_norm * total_violation
        else:
            r_geom_bad_entry = torch.zeros_like(d)

        if self._w_geom_soft_success > 0.0:
            # 3 项 well (d, radial, axis) + 可选 penetration (codex 2026-05-11
            # v4 "clean_dwell"). σ_pen=None 时跟原 3 项行为完全等价.
            # penetration 用 _cached_penetration_max, 在 _create_observation 里写过.
            soft_exponent = (
                - (d_err / self._geom_soft_d_sigma) ** 2
                - (radial_max / self._geom_soft_radial_sigma) ** 2
                - (axis_err / self._geom_soft_axis_sigma) ** 2
            )
            if self._geom_soft_penetration_sigma is not None:
                pen = self._cached_penetration_max
                soft_exponent = soft_exponent - (
                    pen / self._geom_soft_penetration_sigma
                ) ** 2
            soft = torch.exp(soft_exponent)
            r_geom_soft_success = self._w_geom_soft_success * soft
        else:
            r_geom_soft_success = torch.zeros_like(d)

        # Alignment-gated progress (codex 提案). w_progress=0 时 r_geom_progress
        # 全 0, 跟旧 additive 行为完全等价. progress 是 peg 相对起点 d_target_neg
        # 的单调位移, **cap 到当前 d_target_eff** (随 set_geom_epoch ramp 移动),
        # 不是 cap 到 d_target_pos. 否则 ramp_end 之前 agent 也能直接拿到 full
        # progress 奖励, ramp 失效 (codex 2026-05-11 catch). alignment_gate 是
        # Gaussian, 对齐越好越接近 1, 越差越接近 0.
        # progress 公式 (codex 2026-05-11 v2 fix):
        # progress = clamp(d - floor, 0, current_ramp_max - floor)
        # 默认 floor=0 (= hole entrance). 只奖励 peg 真正进入 hole 部分 (d > 0).
        # current_ramp_max = max(floor, d_target_eff): ramp 把 d_target_eff 从
        # d_target_neg 推到 d_target_pos, 但 cap 不下 floor.
        progress_floor = self._geom_progress_floor
        progress_range_full = max(0.0, self._geom_d_target_pos - progress_floor)
        progress_cap = max(
            0.0,
            min(progress_range_full, self._geom_d_target_eff - progress_floor),
        )
        progress = torch.clamp(
            d - progress_floor,
            min=0.0,
            max=progress_cap,
        )
        # alignment_gate / penetration_max 已在 r_d gating 时计算 (v6 refactor),
        # 此处直接复用. 旧版有重复计算, 现去重.
        r_geom_progress = self._w_geom_progress * progress * alignment_gate
        # 可选 soft penalty (默认 0=关). 跟 gate 互补 — gate 让"穿模时拿不到 reward",
        # penalty 让"穿模时被扣 reward". 推荐 SAC: gate + 小 penalty (10-20).
        # Lagrangian SAC: gate + 0 penalty (用 cost 信号代替).
        r_geom_penetration = -self._w_geom_penetration * penetration_max
        # Delta-progress (PBRS, codex 2026-05-11 v3):
        # phi(s) = clean_gate(s) × clamp((d - d_neg) / (d_pos - d_neg), 0, 1).
        # r_advance = w_advance × (phi_t - phi_{t-1}). Episode reset 时 phi_prev
        # 通过 first_step_mask 标记无效化, 该 step Δphi 强制 0.
        d_neg = self._geom_d_target_neg
        d_pos = self._geom_d_target_pos
        normalized_pos = torch.clamp(
            (d - d_neg) / max(1e-6, d_pos - d_neg),
            min=0.0,
            max=1.0,
        )
        phi_t = alignment_gate * normalized_pos
        if self._cached_phi_prev is None:
            # 第一次 reward 计算 (env 启动后), 全 env 视为 first step.
            self._cached_phi_prev = phi_t.detach().clone()
            self._phi_first_step_mask = torch.ones_like(phi_t, dtype=torch.bool)
        if self._phi_first_step_mask is None:
            self._phi_first_step_mask = torch.zeros_like(phi_t, dtype=torch.bool)
        delta_phi = torch.where(
            self._phi_first_step_mask,
            torch.zeros_like(phi_t),
            phi_t - self._cached_phi_prev,
        )
        r_geom_advance = self._w_geom_advance * delta_phi
        # 更新 cache: phi_t 成为下一步的 phi_prev. 同步清 first_step mask.
        self._cached_phi_prev = phi_t.detach().clone()
        self._phi_first_step_mask = torch.zeros_like(phi_t, dtype=torch.bool)

        joint_limit_norm = self._compute_joint_limit_norm(joint_pos)
        action_sq = (self._last_raw_action ** 2).sum(dim=-1)
        home_norm = self._compute_home_norm(joint_pos)
        r_joint_limit = -self._w_joint_limit * joint_limit_norm
        r_action = -self._w_action * action_sq
        r_home = -self._w_home * home_norm

        return {
            "r_geom_d": r_geom_d,
            "r_geom_radial_tip": r_geom_radial_tip,
            "r_geom_radial_max": r_geom_radial_max,
            "r_geom_axis": r_geom_axis,
            "r_geom_soft_success": r_geom_soft_success,
            "r_geom_progress": r_geom_progress,
            "r_geom_penetration": r_geom_penetration,
            "r_geom_advance": r_geom_advance,
            "r_geom_bad_entry": r_geom_bad_entry,
            "r_joint_limit": r_joint_limit,
            "r_action": r_action,
            "r_home": r_home,
        }

    def _compute_geom_reward(self, next_obs):
        comps = self._compute_geom_reward_components(next_obs)
        return sum(comps.values())

    def set_geom_epoch(self, epoch):
        """Actor-relative geometric schedule.

        prepos/preaxis keep d_target fixed at geom_d_target_neg. insert linearly
        moves d_target from geom_d_target_neg to geom_d_target_pos, preserving a
        V-shaped / absolute-error depth objective without an overshoot plateau.
        """
        if self._geom_stage is None:
            return
        if self._geom_stage != "insert":
            self._geom_d_target_eff = self._geom_d_target_neg
            return
        s, e = self._geom_d_target_ramp_start, self._geom_d_target_ramp_end
        if epoch < s:
            t = 0.0
        elif epoch < e:
            t = (epoch - s) / max(e - s, 1)
        else:
            t = 1.0
        self._geom_d_target_eff = (
            (1.0 - t) * self._geom_d_target_neg + t * self._geom_d_target_pos
        )

    def reward(self, obs, action, next_obs, absorbing):
        if self._geom_stage is not None:
            normal = self._compute_geom_reward(next_obs)
        else:
            normal = self._compute_normal_reward(next_obs)
        # 三路选择: collision (硬 absorbing) > hold-N (软 absorbing) > normal
        # geom hold-N 默认关 (terminal_hold_bonus=0), 等价 normal 路径.
        absorbing_r = self._r_min / (1.0 - self.info.gamma)
        r = torch.where(
            self._last_collision_mask,
            torch.full_like(normal, absorbing_r),
            torch.where(
                self._last_hold_done_mask,
                normal + self._terminal_hold_bonus,
                normal,
            ),
        )
        return self._reward_scale * r

    def setup(self, env_indices, obs):
        n = len(env_indices)
        noise = self._initial_joint_noise * (
            2.0 * torch.rand(n, len(ARM_JOINTS), device=self._device) - 1.0
        )
        joint_pos = self._default_joint_pos.unsqueeze(0) + noise
        self._write_data("joint_pos", joint_pos, env_indices)
        self._write_data("joint_vel", torch.zeros_like(joint_pos), env_indices)

        idx_tensor = torch.as_tensor(env_indices, device=self._device, dtype=torch.long)
        self._consecutive_inthresh[idx_tensor] = 0
        # PBRS phi 跨 episode 必须无效化, 否则新 episode 第一步 Δphi 含
        # 上一 episode 末态 phi (~0.9) - 新 episode 初始 phi (~0) = 巨大 spurious
        # delta. 标记 first_step, reward 函数里看到该 mask 时 Δphi=0.
        if self._phi_first_step_mask is None:
            self._phi_first_step_mask = torch.zeros(
                self._n_envs, dtype=torch.bool, device=self._device
            )
        self._phi_first_step_mask[idx_tensor] = True

        # set_joint_positions 只写 DOF buffer; 不 step 的话 BODY_POS / BODY_ROT
        # view 还是 reset 前的值, reset_all 读到的 EE pose 是 stale.
        self._world.step(render=False)

    def _simulation_pre_step(self):
        """每 intermediate step 前注入重力补偿 effort.

        kp=0 velocity drive 只有阻尼项, 对恒定重力是结构性欠阻尼. 真机 iiwa
        velocity mode 底层跑重力补偿, sim 里我们用 G(q) 作为前馈 effort,
        agent 不需要从零学这个 7-DoF 非线性映射.
        """
        tau_g = self._robots.get_generalized_gravity_forces(clone=False)
        self._robots.set_joint_efforts(tau_g[:, self._cj], joint_indices=self._cj)

    # ------------------------------------------------------------------
    # 解析式 peg / hole frame helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _quat_apply(q_wxyz, v):
        """用单位四元数 q (wxyz) 旋转向量 v. 支持 [N,4]×[N,3] 或 [N,4]×[3] 广播.

        返回 [N, 3].
        """
        if v.dim() == 1:
            v = v.unsqueeze(0).expand(q_wxyz.shape[0], -1)
        w = q_wxyz[..., 0]
        x = q_wxyz[..., 1]
        y = q_wxyz[..., 2]
        z = q_wxyz[..., 3]
        vx = v[..., 0]
        vy = v[..., 1]
        vz = v[..., 2]
        tx = 2 * (y * vz - z * vy)
        ty = 2 * (z * vx - x * vz)
        tz = 2 * (x * vy - y * vx)
        rx = vx + w * tx + (y * tz - z * ty)
        ry = vy + w * ty + (z * tx - x * tz)
        rz = vz + w * tz + (x * ty - y * tx)
        return torch.stack([rx, ry, rz], dim=-1)

    @staticmethod
    def _quat_mul(q1_wxyz, q2_wxyz):
        """四元数乘法 (wxyz), 支持广播."""
        if q2_wxyz.dim() == 1:
            q2_wxyz = q2_wxyz.unsqueeze(0).expand(q1_wxyz.shape[0], -1)
        w1, x1, y1, z1 = q1_wxyz.unbind(dim=-1)
        w2, x2, y2, z2 = q2_wxyz.unbind(dim=-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    def _verify_peghole_prims_exist(self):
        """fail fast: peg_tip / hole_entry prim 不在 stage 直接 raise.

        老的 phase 1.5 实现是 print 后继续跑; 现在主线已经是 peg-in-hole, 加载
        无 peg/hole 的 USD 必然导致 _create_observation 用错误的常量 offset
        生成无意义的 frame. 早死早超生.
        """
        try:
            import omni.usd
        except ImportError:
            return  # 单元测试或非 IsaacSim 路径下跳过
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("USD stage 还没初始化 — peg/hole 检查无法执行")
        peg_found = hole_found = False
        for prim in stage.Traverse():
            p = str(prim.GetPath())
            if p.endswith("/Peg/peg_tip"):
                peg_found = True
            if p.endswith("/Hole/hole_entry"):
                hole_found = True
            if peg_found and hole_found:
                return
        raise RuntimeError(
            "stage 里找不到 peg_tip / hole_entry prim. "
            f"usd_path={self._usd_path}\n"
            "M0+ 要求加载带 peg/hole 视觉资产的 USDA. "
            "用 dual_arm_iiwa_with_peghole.usda, 或重跑 build_peghole_usd.py 生成."
        )

    def get_preinsert_frames(self):
        """返回 peg_tip / hole_entry / preinsert_target 的世界帧位姿 (env-local).

        训练循环里 _create_observation 已经算并缓存了同样的量; 但 visualize_*
        的主循环可能在两次 step 之间没经过 _create_observation, cache 会 stale.
        所以这里强制重新查一次 fresh raw obs 再算.

        Returns batched dict (batch=num_envs):
            peg_tip_pos          [N, 3]
            peg_tip_quat         [N, 4]  wxyz, = LeftEE_quat
            peg_axis             [N, 3]  unit, R(LeftEE_quat) · PEG_AXIS_IN_LEFTEE
            peg_axis_quat        [N, 4]  让 +Z apply 后等于 peg_axis 的 quat
            hole_entry_pos       [N, 3]
            hole_entry_quat      [N, 4]  wxyz, = RightEE_quat
            hole_axis            [N, 3]
            hole_axis_quat       [N, 4]
            preinsert_target_pos [N, 3]
            preinsert_target_quat[N, 4]
        """
        raw = self.observation_helper.build_obs(self._task.get_observations(clone=True))
        left_ee = self.observation_helper.get_from_obs(raw, "left_ee_pos")
        right_ee = self.observation_helper.get_from_obs(raw, "right_ee_pos")
        left_quat = self.observation_helper.get_from_obs(raw, "left_ee_rot")
        right_quat = self.observation_helper.get_from_obs(raw, "right_ee_rot")

        peg_tip = left_ee + self._quat_apply(left_quat, self._peg_tip_offset)
        hole_entry = right_ee + self._quat_apply(right_quat, self._hole_entry_offset)
        peg_axis = self._quat_apply(left_quat, self._peg_axis_local)
        hole_axis = self._quat_apply(right_quat, self._hole_axis_local)
        peg_axis = peg_axis / peg_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        hole_axis = hole_axis / hole_axis.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        peg_axis_quat = self._quat_mul(left_quat, self._peg_axis_quat_offset)
        hole_axis_quat = self._quat_mul(right_quat, self._hole_axis_quat_offset)

        preinsert_target_pos = hole_entry + self._preinsert_offset * hole_axis
        preinsert_target_quat = hole_axis_quat.clone()

        return {
            "peg_tip_pos": peg_tip,
            "peg_tip_quat": left_quat,
            "peg_axis_quat": peg_axis_quat,
            "peg_axis": peg_axis,
            "hole_entry_pos": hole_entry,
            "hole_entry_quat": right_quat,
            "hole_axis_quat": hole_axis_quat,
            "hole_axis": hole_axis,
            "preinsert_target_pos": preinsert_target_pos,
            "preinsert_target_quat": preinsert_target_quat,
        }

    def _compute_preinsert_errors(self, frames=None):
        """完整 preinsert 几何误差 — 给 visualize_* 的诊断输出用, 不进 reward.

        训练 reward / is_absorbing 走 _compute_task_errors (从 cached agent obs
        切片读); 这条路径独立查 fresh raw obs, 适合 visualize/diagnose 主循环.
        success_mask 用与训练一致的 (pos<pos_th) ∧ (axis<axis_th) 表达式, axis_th
        默认 inf 时退化为 pos-only.
        """
        if frames is None:
            frames = self.get_preinsert_frames()
        peg_tip = frames["peg_tip_pos"]
        hole_entry = frames["hole_entry_pos"]
        preinsert_target = frames["preinsert_target_pos"]
        peg_axis = frames["peg_axis"]
        hole_axis = frames["hole_axis"]

        pos_vec = peg_tip - preinsert_target
        pos_err = torch.norm(pos_vec, dim=-1)

        axis_dot = torch.sum(peg_axis * hole_axis, dim=-1).clamp(-1.0, 1.0)
        axis_err = 1.0 + axis_dot

        d = peg_tip - hole_entry
        axial_dist = torch.sum(d * hole_axis, dim=-1)
        radial_vec = d - axial_dist.unsqueeze(-1) * hole_axis
        radial_err = torch.norm(radial_vec, dim=-1)

        success_mask = (
            (pos_err < self._preinsert_success_pos_threshold)
            & (axis_err < self._success_axis_threshold)
        )

        return {
            "pos_vec": pos_vec,
            "pos_err": pos_err,
            "axis_dot": axis_dot,
            "axis_err": axis_err,
            "axial_dist": axial_dist,
            "radial_vec": radial_vec,
            "radial_err": radial_err,
            "success_mask": success_mask,
        }

    # ------------------------------------------------------------------
    # Sphere-proxy clearance (PhysX 自碰撞兜底, 全 stage 通用)
    # ------------------------------------------------------------------
    def _build_sphere_proxy_indices(self):
        """从 articulation body_names 解析每侧 sphere proxy 需要的 body 索引.

        构造完成后:
            self._left_arm_joint_idx   [8]   left_arm_link_0..link_7 在 body_names 里的位置
            self._right_arm_joint_idx  [8]
            self._left_ee_proxy_idx    [2]   coupler / hande_link
            self._right_ee_proxy_idx   [2]
            self._proxy_radii_per_side [17]  arm 段 15 球 + EE 段 2 球, 半径来自 env 参数
        """
        body_names = list(self._task.robots.body_names)

        def _resolve_all(names):
            missing = [n for n in names if n not in body_names]
            if missing:
                raise RuntimeError(
                    "build_sphere_proxy_indices: body_names 里缺这些 link: "
                    f"{missing}\navailable: {body_names}"
                )
            return [body_names.index(n) for n in names]

        device = self._device
        self._left_arm_joint_idx = torch.as_tensor(
            _resolve_all(LEFT_ARM_JOINT_BODY_NAMES), device=device, dtype=torch.long
        )
        self._right_arm_joint_idx = torch.as_tensor(
            _resolve_all(RIGHT_ARM_JOINT_BODY_NAMES), device=device, dtype=torch.long
        )
        self._left_ee_proxy_idx = torch.as_tensor(
            _resolve_all(LEFT_EE_PROXY_BODY_NAMES), device=device, dtype=torch.long
        )
        self._right_ee_proxy_idx = torch.as_tensor(
            _resolve_all(RIGHT_EE_PROXY_BODY_NAMES), device=device, dtype=torch.long
        )
        # 每侧 17 球的半径 (顺序: 8 关节 + 7 中点 + 2 EE)
        n_arm = len(LEFT_ARM_JOINT_BODY_NAMES)     # 8
        n_mid = n_arm - 1                          # 7 段中点
        n_ee = len(LEFT_EE_PROXY_BODY_NAMES)       # 2
        radii = torch.empty(n_arm + n_mid + n_ee, device=device, dtype=torch.float32)
        radii[:n_arm + n_mid] = self._proxy_arm_radius
        radii[n_arm + n_mid:] = self._proxy_ee_radius
        self._proxy_radii_per_side = radii         # [17]
        self._n_proxies_per_side = n_arm + n_mid + n_ee

    def _gather_side_proxies(self, body_pos, joint_idx, ee_idx):
        """body_pos: [n_envs, n_bodies, 3] → [n_envs, n_proxy, 3] sphere proxy 球心.

        球心顺序: 8 关节 + 7 中点 + 2 EE, 与 self._proxy_radii_per_side 对齐.
        """
        joints = body_pos[:, joint_idx, :]                   # [n_envs, 8, 3]
        mids = 0.5 * (joints[:, :-1, :] + joints[:, 1:, :])  # [n_envs, 7, 3]
        ee = body_pos[:, ee_idx, :]                          # [n_envs, 2, 3]
        return torch.cat([joints, mids, ee], dim=1)          # [n_envs, n_proxy, 3]

    def _compute_min_clearance(self):
        """sphere-proxy 双臂 clearance, 跨所有 env vectorized.

            clearance_ij = ||c_L_i - c_R_j|| - r_L_i - r_R_j
            min_clearance = clearance.min over (i, j)  → [n_envs]

        Returns:
            min_clearance: [n_envs]   每个 env 当前最小双臂 clearance (m).
                                       <0 表示两侧 sphere proxy 已经穿插.
            info: dict
                "min_pair_left_idx":  [n_envs]  long, 0..18 (球索引 per side)
                "min_pair_right_idx": [n_envs]  long
                "left_proxies":  [n_envs, n_proxy, 3]  球心位置 (env-local world)
                "right_proxies": [n_envs, n_proxy, 3]
        """
        physics_view = self._task.robots._physics_view
        xforms = physics_view.get_link_transforms()       # [n_envs, n_bodies, 7] (xyz+quat)
        xforms_t = torch.as_tensor(xforms, device=self._device, dtype=torch.float32)
        if xforms_t.dim() == 2:
            n_bodies = len(self._task.robots.body_names)
            xforms_t = xforms_t.view(self._n_envs, n_bodies, -1)
        body_pos = xforms_t[..., :3]                      # [n_envs, n_bodies, 3]

        left = self._gather_side_proxies(
            body_pos, self._left_arm_joint_idx, self._left_ee_proxy_idx
        )
        right = self._gather_side_proxies(
            body_pos, self._right_arm_joint_idx, self._right_ee_proxy_idx
        )
        # [n_envs, nL, nR, 3] → [n_envs, nL, nR] center-to-center
        diff = left.unsqueeze(2) - right.unsqueeze(1)
        dist = diff.norm(dim=-1)
        rL = self._proxy_radii_per_side.view(1, -1, 1)    # [1, nL, 1]
        rR = self._proxy_radii_per_side.view(1, 1, -1)    # [1, 1, nR]
        clearance = dist - rL - rR                         # [n_envs, nL, nR]

        n = self._n_proxies_per_side
        flat = clearance.view(self._n_envs, -1)
        min_vals, min_flat = flat.min(dim=1)               # [n_envs]
        left_idx = min_flat // n
        right_idx = min_flat % n

        info = {
            "min_pair_left_idx": left_idx,
            "min_pair_right_idx": right_idx,
            "left_proxies": left,
            "right_proxies": right,
        }
        return min_vals, info
