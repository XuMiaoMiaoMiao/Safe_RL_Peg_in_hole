#!/usr/bin/env bash
#SBATCH -J isaac-compat
#SBATCH -p gpua40
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Job id: $SLURM_JOB_ID"
echo "Node: $(hostname)"

nvidia-smi

"$ROOT_DIR/bin/run_isaac_sim_compatibility_check.sh"
