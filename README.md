# V-Link

## Recovering Lost Visual Representations in Action DiT for Vision-Language-Action Models

**Yehao Lu, Jiarui Yang, Yuning Su, Yufeng Xie, Yu Zhong, Yazhou Zhang,
Haiyu Lan, Kaixiang Lu, Peiwen Lin, Chuang Wang, Zequn Qin, Enyu Li, Xi Li**

Zhejiang University &nbsp;·&nbsp;
The Hong Kong University of Science and Technology (Guangzhou) &nbsp;·&nbsp;
Simon Fraser University &nbsp;·&nbsp; AGIBOT

[Paper](https://arxiv.org/abs/2608.25308) &nbsp;|&nbsp;
[Code](https://github.com/yehaolu/V-Link-open)

This repository is the official implementation of **V-Link**, accompanying the
paper *V-Link: Recovering Lost Visual Representations in Action DiT for
Vision-Language-Action Models*.

## Overview

Dual-system vision-language-action models use a vision-language model for scene
understanding and an action expert for continuous control. Yet the action expert
may not retain full access to the VLM's geometric and semantic information.

V-Link addresses this visual-representation accessibility bottleneck at the
vision-language-to-action interface. It learns complementary visual queries in
the VLM and makes them directly accessible to Action DiT.

- **Spatial Queries** learn 3D geometric cues with training-only depth
  supervision.
- **Semantic Queries** learn 2D semantic cues with training-only segmentation
  supervision.
- A causal attention mask separates both query sets without disturbing the
  native image-text token stream.
- An asymmetric bridge injects semantic context in parallel with image tokens,
  then applies Spatial Queries as dedicated geometric conditioning.
- Auxiliary depth and segmentation heads are removed at inference. Depth,
  segmentation, and camera calibration are not required for deployment.

## Results reported in the paper

V-Link improves the average success rate over the base model on all three
simulation benchmarks evaluated in the paper.

| Benchmark | Base | V-Link | Improvement |
| --- | ---: | ---: | ---: |
| LIBERO | 97.4 | **99.3** | **+1.9** |
| LIBERO-Plus | 43.8 | **75.0** | **+31.2** |
| RoboTwin 2.0 | 38.0 | **56.8** | **+18.8** |

On the AGIBOT A3 Ultra, V-Link reaches 98% success on Autonomous Power-On and
94% on Autonomous Power-Off, improving over the base model by +20 and +24
percentage points. It adds 1.58 ms of inference latency in the paper's setup.

These values are paper results. This public release currently contains the
fixed RoboTwin2.0 training and evaluation path described below.

## Released configuration

The released recipe retains one reproducible V-Link configuration:

- three camera views;
- 75 Spatial Queries and 75 Semantic Queries, each arranged as `3 x 5 x 5`;
- C2 asymmetric query injection with RoPE;
- grouped multi-view depth and segmentation heads;
- action/depth/segmentation loss weights of `1.0 / 0.1 / 0.002` in Full
  training;
- rank-128 VLM LoRA;
- 20k Query Warmup followed by an independent 120k Full stage;
- 15-step dual-arm action chunks.

The paper reports RoboTwin2.0 results on six challenging tasks. The released
client provides a ten-task evaluation suite that contains those six tasks and
four additional tasks.

## Installation

Training and policy serving use the same dependencies as the GR00T N1.6 code
base. In the reference environment, activate the existing Conda environment:

```bash
conda activate gr00t_n1.6
export PYTHON="$(command -v python)"
```

Clone this repository and install it in that environment if it is not already
available on `PYTHONPATH`:

```bash
git clone https://github.com/yehaolu/V-Link-open.git
cd V-Link-open
python -m pip install -e .
```

Download the official GR00T-N1.6-3B checkpoint from
[Hugging Face](https://huggingface.co/nvidia/GR00T-N1.6-3B):

```bash
hf download nvidia/GR00T-N1.6-3B \
  --local-dir checkpoints/GR00T-N1.6-3B
export BASE_MODEL_PATH="$(pwd)/checkpoints/GR00T-N1.6-3B"
```

If the checkpoint is already available elsewhere, set `BASE_MODEL_PATH` to
that local directory instead.

Set up RoboTwin and its evaluation environment by following the official
[RoboTwin repository](https://github.com/robotwin-Platform/robotwin). The
RoboTwin interpreter may remain separate from the training environment.

## Dataset

Set the training dataset and RoboTwin checkout paths:

```bash
export DATASET_PATH="<RoboTwin2 LeRobot dataset directory>"
export ROBOTWIN_ROOT="<RoboTwin repository directory>"
export ROBOTWIN_PYTHON="<RoboTwin environment Python>"
```

Training requires three RGB views plus per-frame depth and semantic targets.
The auxiliary targets are used only during training.

See [the RoboTwin2 guide](examples/RoboTwin2/README.md) for the exact LeRobot
modalities, camera metadata schema, valid labels, and operational overrides.

## Training

Validate the complete dataset before launching a distributed job:

```bash
"${PYTHON}" examples/RoboTwin2/validate_dataset.py "${DATASET_PATH}"
```

Run the 20k Query Warmup:

```bash
bash examples/RoboTwin2/train_query_warmup.sh
```

Warmup writes `outputs/robotwin2_c2_warmup/checkpoint-20000`. Run Full training
from those weights:

```bash
bash examples/RoboTwin2/train_c2_full.sh
```

Full training uses a new optimizer and an independent 120k schedule. Its final
checkpoint is `outputs/robotwin2_c2_full/checkpoint-120000`.

## Evaluation

Start the policy server in the `gr00t_n1.6` environment:

```bash
bash examples/RoboTwin2/eval_server.sh
```

In another shell, activate the RoboTwin environment and run:

```bash
bash examples/RoboTwin2/eval_client.sh
```

The client evaluates `click_alarmclock`, `click_bell`, `move_pillbottle_pad`,
`move_stapler_pad`, `place_bread_skillet`, `place_mouse_pad`,
`place_object_stand`, `place_phone_stand`, `press_stapler`, and `turn_switch`.

## Acknowledgements

V-Link is an independent research project and is not affiliated with or
endorsed by NVIDIA. The implementation uses GR00T N1.6 as its initialization
base and builds on its open research code.

We also thank the authors and maintainers of RoboTwin, LIBERO, LIBERO-Plus, and
the other research projects cited in the paper.

## License

This repository is released under the terms in [LICENSE](LICENSE). Third-party
components remain subject to their respective licenses and notices.

## Citation

If you find this work useful, please cite the
[paper](https://arxiv.org/abs/2608.25308):

```bibtex
@misc{lu2026vlinkrecoveringlostvisual,
  title         = {V-Link: Recovering Lost Visual Representations in Action DiT for Vision-Language-Action Models},
  author        = {Yehao Lu and Jiarui Yang and Yuning Su and Yufeng Xie and Yu Zhong and Yazhou Zhang and Haiyu Lan and Kaixiang Lu and Peiwen Lin and Chuang Wang and Zequn Qin and Enyu Li and Xi Li},
  year          = {2026},
  eprint        = {2608.25308},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2608.25308}
}
```
