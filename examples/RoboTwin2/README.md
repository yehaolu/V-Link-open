# RoboTwin2 fixed C2 workflow

This directory is the complete public RoboTwin2 workflow: validate the dataset,
run the 20k query Warmup, run the 120k Full stage, serve the Full checkpoint,
and evaluate the fixed ten-task suite.

## Prerequisites and data

Run commands from the repository root in an environment with this project and
its GR00T/PyTorch dependencies installed. Training needs a local
GR00T-N1.6-3B model directory and a RoboTwin2 dataset in LeRobot format.
Evaluation also needs a compatible RoboTwin checkout.

Set paths for your environment without editing the scripts:

```bash
export PYTHON="${CONDA_PREFIX}/bin/python"
export BASE_MODEL_PATH="<GR00T-N1.6-3B directory>"
export DATASET_PATH="<RoboTwin2 LeRobot dataset directory>"
export ROBOTWIN_ROOT="<RoboTwin repository directory>"
export ROBOTWIN_PYTHON=python
```

The dataset must provide the three video modalities `image`,
`left_wrist_image`, and `right_wrist_image`; the state and action modalities
`left_gripper`, `right_gripper`, `left_arm_eef`, and `right_arm_eef`; and
`annotation.human.action.task_description`. `meta/info.json` must define
`camera_meta_path`; the path must resolve to one NPZ for every episode using
the loader's `episode_chunk` and `episode_index` placeholders. Each NPZ must
contain `intrinsics` `[frames, 3, 3, 3]`, `cam2world`
`[frames, 3, 4, 4]`, a `depth_u16` target `[frames, 3, height, width]`, and
a segmentation target
(`mesh_segmentation`, `mesh_segmentation_u8`, `mesh_segmentation_u16`,
`segmentation`, or `segmentation_u8`) with the same shape. `view_names` must be
exactly `image`, `left_wrist_image`, `right_wrist_image` in that order, and
`depth_unit` must be `mm` or `m`. These matrices and targets are training-only
loader transport; the inference policy does not accept camera calibration.

## Workflow

Validate first, then run the two training stages:

```bash
"${PYTHON}" examples/RoboTwin2/validate_dataset.py "${DATASET_PATH}"
bash examples/RoboTwin2/train_query_warmup.sh
bash examples/RoboTwin2/train_c2_full.sh
```

By default, `WARMUP_OUTPUT_DIR` is `outputs/robotwin2_c2_warmup`; Warmup writes
`checkpoint-20000` directly inside it, with no experiment-name subdirectory.
Full reads exactly
`${WARMUP_OUTPUT_DIR}/checkpoint-20000`. `FULL_OUTPUT_DIR` defaults to
`outputs/robotwin2_c2_full`; Full writes `checkpoint-120000` directly inside it.
The server uses
`${FULL_OUTPUT_DIR}/checkpoint-120000` unless `CHECKPOINT` is set.

The handoff loads the Warmup checkpoint's model weights into the fixed Full
recipe. It does not carry the Warmup optimizer state or global step into the
new Full output directory: the Full schedule runs for 120k steps. Restarting a
stage with checkpoints already present in its own stage output directory resumes the
latest Trainer checkpoint there.

Start the server (the default checkpoint is the Full 120k checkpoint):

```bash
bash examples/RoboTwin2/eval_server.sh
```

In another shell, run the client:

```bash
bash examples/RoboTwin2/eval_client.sh
```

The client evaluates `click_alarmclock`, `click_bell`, `move_pillbottle_pad`,
`move_stapler_pad`, `place_bread_skillet`, `place_mouse_pad`,
`place_object_stand`, `place_phone_stand`, `press_stapler`, and `turn_switch`.

## Fixed method and operational overrides

The method is fixed in `train.py`: three cameras; 75 action queries and 75
segmentation queries arranged as `3 x 5 x 5`; the C2 bridge with RoPE; grouped
depth and segmentation heads; Full-stage action/depth/segmentation loss weights
of `1.0 / 0.1 / 0.002`; VLM LoRA rank 128; and fixed Warmup/Full schedules of
20k/120k steps. Warmup saves at step 20k; Full saves every 40k, including its
step-120k final checkpoint. Warmup intentionally sets the action loss to zero
and freezes the action path while learning the query auxiliaries; Full enables
the action path with the stated weights. These settings, the stage ordering,
camera set, task set, task config, and chunk execution mode are not script
overrides.

Permitted environment overrides are operational only. Training accepts
`BASE_MODEL_PATH`, `DATASET_PATH`, `PYTHON`, `WARMUP_OUTPUT_DIR`,
`FULL_OUTPUT_DIR`, `NUM_GPUS`, `GPU_IDS`, `DATALOADER_NUM_WORKERS`, `SEED`,
`GLOBAL_BATCH_SIZE`, `LEARNING_RATE`, and `MASTER_PORT`. Serving and evaluation
accept `FULL_OUTPUT_DIR`, `CHECKPOINT`,
`PYTHON`, `HOST`, `PORT`, `GPU`, `ROBOTWIN_ROOT`, `ROBOTWIN_PYTHON`, `SEED`,
`TEST_NUM`, `RUN_NAME`, and `LOG_DIR`.
