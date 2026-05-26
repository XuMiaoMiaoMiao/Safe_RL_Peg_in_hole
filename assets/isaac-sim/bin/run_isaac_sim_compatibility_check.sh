#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_SIF="${IMAGE_SIF:-$ROOT_DIR/pkg/isaac-sim.sif}"

if [[ ! -f "$IMAGE_SIF" ]]; then
  echo "Missing image: $IMAGE_SIF" >&2
  echo "Run $ROOT_DIR/bin/pull_isaac_sim.sh first." >&2
  exit 1
fi

export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export APPTAINERENV_ACCEPT_EULA="$ACCEPT_EULA"
export APPTAINERENV_PRIVACY_CONSENT="$PRIVACY_CONSENT"

exec apptainer exec \
  --nv \
  --containall \
  "$IMAGE_SIF" \
  /isaac-sim/isaac-sim.compatibility_check.sh --/app/quitAfter=10 --no-window
