#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_URI="${IMAGE_URI:-docker://nvcr.io/nvidia/isaac-sim:5.1.0}"
OUTPUT_SIF="${OUTPUT_SIF:-$ROOT_DIR/pkg/isaac-sim.sif}"

mkdir -p "$ROOT_DIR/pkg"

echo "Pulling Isaac Sim image"
echo "  URI: $IMAGE_URI"
echo "  SIF: $OUTPUT_SIF"

apptainer pull --force "$OUTPUT_SIF" "$IMAGE_URI"

echo "Done"
