#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_SIF="${IMAGE_SIF:-$ROOT_DIR/pkg/isaac-sim.sif}"

if [[ ! -f "$IMAGE_SIF" ]]; then
  echo "Missing image: $IMAGE_SIF" >&2
  echo "Run $ROOT_DIR/bin/pull_isaac_sim.sh first." >&2
  exit 1
fi

mkdir -p \
  "$ROOT_DIR/cache/main" \
  "$ROOT_DIR/cache/computecache" \
  "$ROOT_DIR/logs" \
  "$ROOT_DIR/config" \
  "$ROOT_DIR/data"

export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export APPTAINERENV_ACCEPT_EULA="$ACCEPT_EULA"
export APPTAINERENV_PRIVACY_CONSENT="$PRIVACY_CONSENT"

exec apptainer exec \
  --nv \
  --containall \
  --bind "$ROOT_DIR/cache/main:/isaac-sim/.cache" \
  --bind "$ROOT_DIR/cache/computecache:/isaac-sim/.nv/ComputeCache" \
  --bind "$ROOT_DIR/logs:/isaac-sim/.nvidia-omniverse/logs" \
  --bind "$ROOT_DIR/config:/isaac-sim/.nvidia-omniverse/config" \
  --bind "$ROOT_DIR/data:/isaac-sim/.local/share/ov/data" \
  "$IMAGE_SIF" \
  /isaac-sim/runheadless.sh -v
