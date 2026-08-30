#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

: "${ROBOTWIN_ROOT:?Set ROBOTWIN_ROOT to the RoboTwin repository}"

ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5710}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
TEST_NUM="${TEST_NUM:-100}"
RUN_NAME="${RUN_NAME:-robotwin2_c2_full}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/robotwin2_eval/${RUN_NAME}}"

if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
  echo "RoboTwin repository does not exist: ${ROBOTWIN_ROOT}" >&2
  exit 1
fi

TASKS=(
  move_pillbottle_pad
  move_stapler_pad
  place_bread_skillet
  place_phone_stand
  press_stapler
  turn_switch
)

mkdir -p "${LOG_DIR}"
for task in "${TASKS[@]}"; do
  (
    cd "${ROBOTWIN_ROOT}"
    export CUDA_VISIBLE_DEVICES="${GPU}"
    "${ROBOTWIN_PYTHON}" -u script/eval_policy.py \
      --config policy/GR00T_EEF/deploy_policy.yml \
      --overrides \
      --task_name "${task}" \
      --task_config demo_clean \
      --ckpt_setting "${RUN_NAME}" \
      --seed "${SEED}" \
      --policy_name GR00T_EEF \
      --gr00t_server_host "${HOST}" \
      --gr00t_server_port "${PORT}" \
      --e2e_repo_path "${REPO_ROOT}" \
      --chunk_exec full \
      --test_num "${TEST_NUM}"
  ) 2>&1 | tee "${LOG_DIR}/${task}.log"
done
