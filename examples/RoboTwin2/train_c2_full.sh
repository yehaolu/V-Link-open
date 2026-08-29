#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${DATASET_PATH:?Set DATASET_PATH to the RoboTwin2 LeRobot dataset}"

WARMUP_OUTPUT_DIR="${WARMUP_OUTPUT_DIR:-${REPO_ROOT}/outputs/robotwin2_c2_warmup}"
WARMUP_CHECKPOINT="${WARMUP_OUTPUT_DIR}/checkpoint-20000"
FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-${REPO_ROOT}/outputs/robotwin2_c2_full}"
NUM_GPUS="${NUM_GPUS:-4}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
SEED="${SEED:-0}"
MASTER_PORT="${MASTER_PORT:-29629}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
PYTHON="${PYTHON:-python}"

WARMUP_OUTPUT_CANONICAL="$(realpath -m -- "${WARMUP_OUTPUT_DIR}")"
FULL_OUTPUT_CANONICAL="$(realpath -m -- "${FULL_OUTPUT_DIR}")"
if [[ "${WARMUP_OUTPUT_CANONICAL}" == "${FULL_OUTPUT_CANONICAL}" ]]; then
  echo "FULL_OUTPUT_DIR must be different from WARMUP_OUTPUT_DIR; both resolve to: ${FULL_OUTPUT_CANONICAL}" >&2
  echo "Full training auto-resumes trainer state from its output directory. Choose a dedicated FULL_OUTPUT_DIR." >&2
  exit 1
fi

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

if [[ ! -d "${WARMUP_CHECKPOINT}" ]]; then
  echo "Warmup checkpoint does not exist: ${WARMUP_CHECKPOINT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# "${PYTHON}" "${REPO_ROOT}/examples/RoboTwin2/validate_dataset.py" "${DATASET_PATH}"

exec "${PYTHON}" -m torch.distributed.run \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  "${REPO_ROOT}/examples/RoboTwin2/train.py" \
  --stage full \
  --base-model-path "${WARMUP_CHECKPOINT}" \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${FULL_OUTPUT_DIR}" \
  --num-gpus "${NUM_GPUS}" \
  --seed "${SEED}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --learning-rate "${LEARNING_RATE}" \
  --dataloader-num-workers "${DATALOADER_NUM_WORKERS}"
