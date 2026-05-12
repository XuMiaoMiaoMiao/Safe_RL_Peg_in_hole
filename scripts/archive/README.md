# scripts/archive/ — 一次性诊断脚本 + v4 baseline 还原源

主线 (train/eval/visualize/_eval_utils) 不依赖这里。**保留是为了 USD 资产换代复用,
或者把已删除的 v4 reward 路径恢复成可运行 baseline**。

| 脚本 / 文件 | 写它的 stage | 验证了什么 / 用途 | 是否可直接运行 |
|---|---|---|---|
| `check_peghole_asset.py` | M0 期 | 新 composed USDA 是 articulation 层 no-op (diff DOF / G(q) / default_pos) | ✓ |
| `dryrun_stage3_v4.py` | Stage 3 v4 期 | v4 reward 各分量 mean/std/min/max dry-run | ✗ (v4 env 方法已删, 见 v4_baseline.py 还原) |
| `sanity_eval_stage3.py` | Stage 3 v4 期 | 钉死 `d = -axial_dist` sign convention + collision count | ✗ (依赖 v4 stage3 path) |
| `plot_geom_reward_landscape.py` | geom v1-v7 reward 设计期 | 纯 numpy 算 geom reward 公式, 输出 HTML SVG landscape | ✓ (无 IsaacSim 依赖) |
| `plot_geom_reward_ideal.py` | geom v1-v7 reward 设计期 | 理想 policy 轨迹下 geom reward 各分量曲线 PNG | ✓ (无 IsaacSim 依赖) |
| `plot_geom_gated_compare.py` | codex gated-progress 提案 | 数值验证 alignment-gated vs additive progress 的 stay/advance 局部排序 | ✓ (无 IsaacSim 依赖) |
| `v4_baseline.py` | 2026-05-12 v4 删除时 | Stage 3 v4 完整代码快照 (env 方法 + CLI args + state init + dispatch + metrics) | ✗ (snapshot 不是可执行 module, 是 paste-back 源) |
| `legacy_stage1_stage2.md` | pre-geom_stage 训练期 | 旧 Stage 1/2 球形 pos_err / axis cliff 配方与结果记录 | 文档 |

## v4 reward 还原流程

主代码里 `--stage3` / `_compute_stage3_*` / `set_stage3_epoch` /
`compute_stage3_metrics` 在 2026-05-12 全部移除. 想跑 v4 baseline 实验时:

1. 打开 `v4_baseline.py`, 按头部 "RESTORATION RECIPE" 把 9 个 section 分别 paste
   回 env / train_sac / eval_sac / visualize_policy / _eval_utils
2. (可选) 重命名 `_peg_sample_offsets` → `_stage3_peg_sample_offsets` (或保留新名
   让 v4 reward 读 `self._peg_sample_offsets`)
3. `python -m py_compile envs/*.py scripts/*.py` 验证
4. 跑 dryrun 把 reward 各分量 sanity check 一遍: `python scripts/archive/dryrun_stage3_v4.py`
5. 用 `v4_baseline.py` section 9 里的旧 Stage 3 v4 训练命令跑 baseline

## 何时再跑

- 改 USD 资产 (重新生成 `dual_arm_iiwa_with_peghole.usda`) → `check_peghole_asset.py`
  验证 articulation 物理 no-op
- 想再设计新 reward 配方 (geom 外) → 三个 `plot_geom_*.py` 是公式 sanity 模板,
  不依赖 IsaacSim
- v4 baseline 对照实验 → 按 v4_baseline.py 还原,然后跑 dryrun_stage3_v4 + sanity_eval_stage3

CLI 默认路径假设你 `cd /home/miao/bimanual_peghole` 后 `python scripts/archive/<name>.py`.
