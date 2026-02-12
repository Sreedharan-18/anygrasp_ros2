#!/usr/bin/env bash
set -euo pipefail

# Run from: .../anygrasp_sdk/grasp_detection
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRASPNET_ROOT="${ROOT_DIR}/data/graspnet"

# === Config you typically change ===
SCENE_L=160
SCENE_R=190            # exclusive
CAMERA="realsense"
CKPT="${ROOT_DIR}/log/checkpoint_detection.tar"
DUMP_DIR="${ROOT_DIR}/dumps_anygrasp_grasps"

# === "Good AP" defaults for 8GB VRAM + collision on ===
MAX_POINTS=40000       
Z_MIN=0.02
Z_MAX=1.50
TOPK_SAVE=50
CPU_THREADS=1

# Frames/views
START_FRAME=0
END_FRAME=256

# Optional: reproducibility
SEED=123

echo "[INFO] GRASPNET_ROOT=${GRASPNET_ROOT}"
echo "[INFO] CKPT=${CKPT}"
echo "[INFO] scenes=[${SCENE_L},${SCENE_R}) camera=${CAMERA}"
echo "[INFO] dump_dir=${DUMP_DIR}"

# Hard checks
test -f "${CKPT}" || { echo "[ERROR] Missing checkpoint: ${CKPT}"; exit 1; }
test -f "${GRASPNET_ROOT}/scenes/scene_$(printf "%04d" ${SCENE_L})/${CAMERA}/camK.npy" || {
  echo "[ERROR] Missing camK.npy for first scene. Check GRASPNET_ROOT and CAMERA."
  exit 1
}

python3 test_graspnet.py \
  --mode eval \
  --graspnet-root "${GRASPNET_ROOT}" \
  --camera "${CAMERA}" \
  --scene-l "${SCENE_L}" --scene-r "${SCENE_R}" \
  --checkpoint-path "${CKPT}" \
  --dump-dir "${DUMP_DIR}" \
  --cpu-threads "${CPU_THREADS}" \
  --seed "${SEED}" \
  --max-points "${MAX_POINTS}" \
  --z-min "${Z_MIN}" --z-max "${Z_MAX}" \
  --topk-save "${TOPK_SAVE}" \
  --start-frame "${START_FRAME}" --end-frame "${END_FRAME}"
