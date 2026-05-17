#!/usr/bin/env bash
# DEPRECATED: do not use. This was a manual-parameter draft before the
# Lagrangian Hydra YAMLs were inspected. Use conf/experiment/lag_stage*.yaml
# via scripts/train_hydra.py instead.
set -euo pipefail

# Full-chain Lagrangian SAC comparison against the current successful SAC chain.
#
# Fairness rule:
#   - Stage 1g Lagrangian uses the same stage/seed/horizon/steps/reward params
#     as Stage 1g SAC, plus only Lagrangian cost/lambda params.
#   - Stage 2g Lagrangian warm-starts from the Lagrangian Stage 1g best_hold,
#     mirroring SAC Stage 2g warm-starting from SAC Stage 1g best_hold.
#   - Stage 3g Lagrangian warm-starts from the Lagrangian Stage 2g best_hold,
#     mirroring SAC Stage 3g warm-starting from SAC Stage 2g best_hold.
#
# This compares the full algorithmic pipeline, not a SAC-assisted Lagrangian
# fine-tune from already-successful SAC intermediate checkpoints.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
WANDB_GROUP="geom_lagrangian_full_chain_compare_${STAMP}"
RUN_LOG_DIR="results/lagrangian_full_chain_compare/${STAMP}"
mkdir -p "${RUN_LOG_DIR}"

latest_lag_dir() {
  find results/checkpoints_lag -mindepth 2 -maxdepth 2 -type d -printf '%T@ %p\n' \
    | sort -nr \
    | head -1 \
    | cut -d' ' -f2-
}

save_latest_lag_ckpt() {
  local tag="$1"
  local raw_dir
  local saved_dir
  raw_dir="$(latest_lag_dir)"
  saved_dir="results/checkpoints/saved/${tag}"
  mkdir -p "${saved_dir}"
  cp "${raw_dir}/best_hold.msh" "${saved_dir}/best_hold.msh"
  cp "${raw_dir}/best_agent.msh" "${saved_dir}/best_agent.msh"
  cp "${raw_dir}/final_agent.msh" "${saved_dir}/final_agent.msh"
  printf '%s\n' "${raw_dir}" > "${saved_dir}/raw_checkpoint_dir.txt"
  echo "${saved_dir}"
}

COMMON_LAG_ARGS=(
  --cost_signal collision
  --cost_limit_per_ep 0.05
  --lambda_update_mode rollout_episode_rate
  --lr_lambda 1e-3
  --lambda_min 0.05
  --lambda_max 100
  --init_log_lambda 0.0
  --actor_grad_clip 1.0
)

COMMON_HOME=(--home_weights 1,1,1,1,0.75,0.5,0.5)

echo "[0/6] py_compile sanity"
python -m py_compile \
  envs/dual_arm_peg_hole_env.py \
  envs/dual_arm_peg_hole_cost_env.py \
  algorithm/lagrangian_sac.py \
  scripts/train_sac_lagrangian.py \
  scripts/eval_sac.py \
  scripts/record_geom_video.py

echo "[1/6] Lagrangian Stage 1g prepos, same params as SAC Stage 1g"
S1_TAG="LagSAC_Stage1g_h100_full_ep_cold_seed0_${STAMP}"
python scripts/train_sac_lagrangian.py \
  --geom_stage prepos \
  --geom_radial_sat 1.5 --geom_d_sat 0.30 \
  --rew_geom_d 8.0 --rew_geom_radial_tip 8.0 \
  "${COMMON_LAG_ARGS[@]}" \
  --horizon 100 --n_epochs 120 --n_steps_per_epoch 1600 \
  --num_envs 16 --n_eval_episodes 16 \
  --critic_warmup_transitions 10000 \
  --terminal_hold_bonus 0 --hold_success_steps 10 \
  --rew_home 0.00075 "${COMMON_HOME[@]}" \
  --lr_actor 7e-5 --lr_critic 3e-4 --lr_alpha 3e-4 \
  --alpha_max 0.15 --target_entropy -7 \
  --seed 0 \
  --wandb_run_name "${S1_TAG}" --wandb_group "${WANDB_GROUP}" \
  2>&1 | tee "${RUN_LOG_DIR}/${S1_TAG}.log"
S1_SAVED="$(save_latest_lag_ckpt "${S1_TAG}")"

echo "[2/6] Lagrangian Stage 2g preaxis, same params as SAC Stage 2g"
S2_TAG="LagSAC_Stage2g_preaxis_h150_from_lag_stage1_seed0_${STAMP}"
python scripts/train_sac_lagrangian.py \
  --geom_stage preaxis \
  --load_agent "${S1_SAVED}/best_hold.msh" \
  --actor_only_warmstart \
  --critic_warmup_transitions 50000 \
  --geom_d_target_neg -0.08 \
  --geom_d_sat 0.30 --geom_radial_sat 1.0 \
  --geom_d_th 0.020 --geom_r_tip_th 0.020 \
  --geom_r_max_th 0.025 --geom_axis_th 0.300 \
  --rew_geom_d 8.0 --rew_geom_radial_tip 2.0 \
  --rew_geom_radial_max 5.0 --rew_geom_axis 1.2 \
  "${COMMON_LAG_ARGS[@]}" \
  --horizon 150 --n_epochs 120 --n_steps_per_epoch 2400 \
  --num_envs 16 --n_eval_episodes 16 \
  --hold_success_steps 10 --terminal_hold_bonus 0 \
  --rew_home 0.00075 "${COMMON_HOME[@]}" \
  --lr_actor 3e-5 --lr_critic 3e-4 --lr_alpha 3e-4 \
  --alpha_max 0.05 --target_entropy -7 \
  --seed 0 \
  --wandb_run_name "${S2_TAG}" --wandb_group "${WANDB_GROUP}" \
  2>&1 | tee "${RUN_LOG_DIR}/${S2_TAG}.log"
S2_SAVED="$(save_latest_lag_ckpt "${S2_TAG}")"

echo "[3/6] Lagrangian Stage 3g insert, same params as SAC Stage 3g"
S3_TAG="LagSAC_Stage3g_insert_h200_from_lag_stage2_seed1_${STAMP}"
python scripts/train_sac_lagrangian.py \
  --geom_stage insert \
  --load_agent "${S2_SAVED}/best_hold.msh" \
  --actor_only_warmstart \
  --critic_warmup_transitions 30000 \
  --exclude_ee_from_physx_self_collision \
  --geom_d_target_neg -0.08 --geom_d_target_pos 0.03 \
  --geom_d_target_ramp_start 0 --geom_d_target_ramp_end 20 \
  --geom_progress_floor 0.0 \
  --geom_d_sat 0.30 --geom_radial_sat 1.0 \
  --geom_d_th 0.020 --geom_r_tip_th 0.015 \
  --geom_r_max_th 0.025 --geom_axis_th 0.300 \
  --geom_insert_d_ins 0.025 --geom_insert_r_max_th 0.025 \
  --geom_pen_th 0.001 \
  --rew_geom_d 1.0 \
  --rew_geom_radial_tip 0 \
  --rew_geom_radial_max 5.0 \
  --rew_geom_axis 1.0 \
  --rew_geom_progress 0.0 \
  --rew_geom_advance 25.0 \
  --geom_gate_radial_sigma 0.025 \
  --geom_gate_axis_sigma 0.30 \
  --rew_geom_penetration 20.0 \
  --geom_gate_penetration_sigma 0.005 \
  --rew_geom_soft_success 1.25 \
  --geom_soft_d_sigma 0.018 \
  --geom_soft_radial_sigma 0.009 \
  --geom_soft_axis_sigma 0.15 \
  --geom_soft_penetration_sigma 0.0025 \
  --geom_d_gate_mode off \
  --rew_geom_bad_entry 0.30 \
  --geom_bad_entry_radial_safe 0.014 \
  --geom_bad_entry_axis_safe 0.10 \
  --geom_bad_entry_pen_safe 0.00075 \
  "${COMMON_LAG_ARGS[@]}" \
  --horizon 200 --n_epochs 80 --n_steps_per_epoch 3200 \
  --num_envs 16 --n_eval_episodes 16 \
  --hold_success_steps 10 --terminal_hold_bonus 0 \
  --rew_home 0.001 "${COMMON_HOME[@]}" \
  --lr_actor 1e-5 --lr_critic 3e-4 --lr_alpha 3e-4 \
  --alpha_max 0.015 --target_entropy -7 \
  --seed 1 \
  --wandb_run_name "${S3_TAG}" --wandb_group "${WANDB_GROUP}" \
  2>&1 | tee "${RUN_LOG_DIR}/${S3_TAG}.log"
S3_SAVED="$(save_latest_lag_ckpt "${S3_TAG}")"

echo "[4/6] Eval best_hold checkpoints with the same eval conditions as SAC"
python scripts/eval_sac.py \
  --agent_path "${S1_SAVED}/best_hold.msh" \
  --geom_stage prepos \
  --geom_eval_epoch 0 \
  --horizon 100 \
  --num_envs 16 --n_episodes 16 \
  --hold_success_steps 10 --terminal_hold_bonus 0 \
  --rew_home 0.00075 "${COMMON_HOME[@]}" \
  --headless \
  2>&1 | tee "${S1_SAVED}/eval_best_hold.txt"

python scripts/eval_sac.py \
  --agent_path "${S2_SAVED}/best_hold.msh" \
  --geom_stage preaxis \
  --geom_eval_epoch 0 \
  --geom_d_target_neg -0.08 \
  --geom_d_sat 0.30 --geom_radial_sat 1.0 \
  --geom_d_th 0.020 --geom_r_tip_th 0.020 \
  --geom_r_max_th 0.025 --geom_axis_th 0.300 \
  --horizon 150 \
  --num_envs 16 --n_episodes 16 \
  --hold_success_steps 10 --terminal_hold_bonus 0 \
  --rew_home 0.00075 "${COMMON_HOME[@]}" \
  --headless \
  2>&1 | tee "${S2_SAVED}/eval_best_hold.txt"

python scripts/eval_sac.py \
  --agent_path "${S3_SAVED}/best_hold.msh" \
  --geom_stage insert \
  --geom_eval_epoch 20 \
  --exclude_ee_from_physx_self_collision \
  --geom_d_target_neg -0.08 --geom_d_target_pos 0.03 \
  --geom_d_target_ramp_start 0 --geom_d_target_ramp_end 20 \
  --geom_progress_floor 0.0 \
  --geom_d_sat 0.30 --geom_radial_sat 1.0 \
  --geom_d_th 0.020 --geom_r_tip_th 0.015 \
  --geom_r_max_th 0.025 --geom_axis_th 0.300 \
  --geom_insert_d_ins 0.025 --geom_insert_r_max_th 0.025 \
  --geom_pen_th 0.001 \
  --rew_geom_d 1.0 \
  --rew_geom_radial_tip 0 \
  --rew_geom_radial_max 5.0 \
  --rew_geom_axis 1.0 \
  --rew_geom_progress 0.0 \
  --rew_geom_advance 25.0 \
  --geom_gate_radial_sigma 0.025 \
  --geom_gate_axis_sigma 0.30 \
  --rew_geom_penetration 20.0 \
  --geom_gate_penetration_sigma 0.005 \
  --rew_geom_soft_success 1.25 \
  --geom_soft_d_sigma 0.018 \
  --geom_soft_radial_sigma 0.009 \
  --geom_soft_axis_sigma 0.15 \
  --geom_soft_penetration_sigma 0.0025 \
  --geom_d_gate_mode off \
  --rew_geom_bad_entry 0.30 \
  --geom_bad_entry_radial_safe 0.014 \
  --geom_bad_entry_axis_safe 0.10 \
  --geom_bad_entry_pen_safe 0.00075 \
  --cost_signal collision \
  --horizon 200 \
  --num_envs 16 --n_episodes 16 \
  --hold_success_steps 10 --terminal_hold_bonus 0 \
  --rew_home 0.001 "${COMMON_HOME[@]}" \
  --headless \
  2>&1 | tee "${S3_SAVED}/eval_best_hold.txt"

echo "[5/6] Record one best_hold video per Lagrangian stage"
VIDEO_DIR="results/videos/LagSAC_full_chain_best_hold_${STAMP}"
mkdir -p "${VIDEO_DIR}"

python scripts/record_geom_video.py \
  --agent_path "${S1_SAVED}/best_hold.msh" \
  --output_dir "${VIDEO_DIR}" \
  --tag stage1g_lag_prepos_best_hold \
  --geom_stage prepos \
  --geom_eval_epoch 0 \
  --horizon 100 \
  --num_envs 2 --viz_env_idx 0 --n_episodes 1 \
  --capture_backend viewport \
  --final_hold_seconds 2.0 \
  --rew_home 0.00075 "${COMMON_HOME[@]}"

python scripts/record_geom_video.py \
  --agent_path "${S2_SAVED}/best_hold.msh" \
  --output_dir "${VIDEO_DIR}" \
  --tag stage2g_lag_preaxis_best_hold \
  --geom_stage preaxis \
  --geom_eval_epoch 0 \
  --geom_d_target_neg -0.08 \
  --horizon 150 \
  --num_envs 2 --viz_env_idx 0 --n_episodes 1 \
  --capture_backend viewport \
  --final_hold_seconds 2.0 \
  --rew_home 0.00075 "${COMMON_HOME[@]}"

python scripts/record_geom_video.py \
  --agent_path "${S3_SAVED}/best_hold.msh" \
  --output_dir "${VIDEO_DIR}" \
  --tag stage3g_lag_insert_best_hold \
  --geom_stage insert \
  --geom_eval_epoch 20 \
  --exclude_ee_from_physx_self_collision \
  --horizon 200 \
  --num_envs 2 --viz_env_idx 0 --n_episodes 1 \
  --capture_backend viewport \
  --final_hold_seconds 2.0 \
  --rew_home 0.001 "${COMMON_HOME[@]}"

echo "[6/6] Done"
echo "W&B group: ${WANDB_GROUP}"
echo "Stage 1 saved: ${S1_SAVED}"
echo "Stage 2 saved: ${S2_SAVED}"
echo "Stage 3 saved: ${S3_SAVED}"
echo "Videos: ${VIDEO_DIR}"
echo "Logs: ${RUN_LOG_DIR}"
