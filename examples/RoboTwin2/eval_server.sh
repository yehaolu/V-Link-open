#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-${REPO_ROOT}/outputs/robotwin2_c2_full}"
CHECKPOINT="${CHECKPOINT:-${FULL_OUTPUT_DIR}/checkpoint-120000}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5710}"
GPU="${GPU:-0}"
PYTHON="${PYTHON:-python}"

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

exec "${PYTHON}" -u "${REPO_ROOT}/gr00t/eval/run_gr00t_server.py" \
  --model-path "${CHECKPOINT}" \
  --embodiment-tag NEW_EMBODIMENT \
  --host "${HOST}" \
  --port "${PORT}" \
  --device cuda:0 \
  --use-sim-policy-wrapper
