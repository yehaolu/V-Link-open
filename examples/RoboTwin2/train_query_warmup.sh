#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${BASE_MODEL_PATH:?Set BASE_MODEL_PATH to GR00T-N1.6-3B}"
: "${DATASET_PATH:?Set DATASET_PATH to the RoboTwin2 LeRobot dataset}"

WARMUP_OUTPUT_DIR="${WARMUP_OUTPUT_DIR:-${REPO_ROOT}/outputs/robotwin2_c2_warmup}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-29628}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
PYTHON="${PYTHON:-python}"

if ! [[ "${NUM_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NUM_GPUS must be a positive integer, got: ${NUM_GPUS}" >&2
  exit 1
fi
if ! [[ "${GPU_IDS}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_IDS must be comma-separated non-negative integer IDs with no empty elements, got: ${GPU_IDS}" >&2
  exit 1
fi
IFS=',' read -r -a GPU_ID_LIST <<< "${GPU_IDS}"
if [[ "${#GPU_ID_LIST[@]}" -ne "${NUM_GPUS}" ]]; then
  echo "NUM_GPUS=${NUM_GPUS}, but GPU_IDS contains ${#GPU_ID_LIST[@]} IDs: ${GPU_IDS}" >&2
  exit 1
fi
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_ID_LIST[@]}"; do
  canonical_gpu_id="${gpu_id#"${gpu_id%%[!0]*}"}"
  canonical_gpu_id="${canonical_gpu_id:-0}"
  if [[ -n "${SEEN_GPU_IDS[${canonical_gpu_id}]+present}" ]]; then
    echo "GPU_IDS must contain unique IDs, got duplicate ID ${gpu_id}: ${GPU_IDS}" >&2
    exit 1
  fi
  SEEN_GPU_IDS["${canonical_gpu_id}"]=1
done

if [[ ! -d "${BASE_MODEL_PATH}" ]]; then
  echo "Base model directory does not exist: ${BASE_MODEL_PATH}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# "${PYTHON}" "${REPO_ROOT}/examples/RoboTwin2/validate_dataset.py" "${DATASET_PATH}"

exec "${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/examples/RoboTwin2/train.py" \
  --stage warmup \
  --base-model-path "${BASE_MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${WARMUP_OUTPUT_DIR}" \
  --num-gpus "${NUM_GPUS}" \
  --seed "${SEED}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
