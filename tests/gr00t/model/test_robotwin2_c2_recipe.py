from dataclasses import fields
import os
from pathlib import Path
import shlex
import subprocess

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "RoboTwin2"


def test_recipe_fixes_the_selected_75_query_c2_topology():
    from examples.RoboTwin2.train import apply_c2_recipe
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

    config = Gr00tN1d6Config()
    apply_c2_recipe(config, stage="full")

    assert config.action_query_num_tokens == 75
    assert config.action_query_grid_size == (5, 5)
    assert config.segmentation_query_num_tokens == 75
    assert config.segmentation_query_grid_size == (5, 5)
    assert config.action_query_depth_num_cameras == 3
    assert config.action_query_segmentation_num_cameras == 3
    assert config.action_query_depth_head_type == "grouped_action_query_depth_head"
    assert (
        config.action_query_segmentation_head_type
        == "grouped_action_query_segmentation_head"
    )
    assert config.use_action_query_depth is True
    assert config.use_action_query_segmentation is True
    assert config.use_action_query_bridge is True
    assert config.action_query_bridge_context_mode == "c2"
    assert config.action_query_bridge_use_rope is True
    assert config.action_query_bridge_weight == 1.0
    assert config.disentangle_action_segmentation_queries is True


@pytest.mark.parametrize(
    ("stage", "action_loss", "tune_action", "freeze_bridge"),
    [
        ("warmup", 0.0, False, True),
        ("full", 1.0, True, False),
    ],
)
def test_recipe_encodes_the_two_training_stages(
    stage: str,
    action_loss: float,
    tune_action: bool,
    freeze_bridge: bool,
):
    from examples.RoboTwin2.train import apply_c2_recipe
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

    config = Gr00tN1d6Config()
    apply_c2_recipe(config, stage=stage)

    assert config.action_loss_weight == action_loss
    assert config.action_query_depth_loss_weight == 0.1
    assert config.action_query_segmentation_loss_weight == 0.002
    assert config.tune_projector is tune_action
    assert config.tune_diffusion_model is tune_action
    assert config.tune_vlln is tune_action
    assert config.freeze_action_query_bridge is freeze_bridge
    assert config.use_vlm_lora is True
    assert config.vlm_lora_rank == 128
    assert config.freeze_vlm_lora is False


def test_recipe_rejects_unknown_stage():
    from examples.RoboTwin2.train import apply_c2_recipe
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

    with pytest.raises(ValueError, match="warmup or full"):
        apply_c2_recipe(Gr00tN1d6Config(), stage="unsupported")


def test_build_config_propagates_dataloader_worker_count(tmp_path: Path):
    from examples.RoboTwin2.train import RoboTwin2TrainConfig, build_config

    args = RoboTwin2TrainConfig(
        stage="warmup",
        base_model_path="base-model",
        dataset_path=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        dataloader_num_workers=13,
    )

    config = build_config(args)

    assert config.training.dataloader_num_workers == 13


def test_build_config_uses_eagle_required_bfloat16(tmp_path: Path):
    from examples.RoboTwin2.train import RoboTwin2TrainConfig, build_config

    config = build_config(
        RoboTwin2TrainConfig(
            stage="warmup",
            base_model_path="base-model",
            dataset_path=tmp_path / "dataset",
            output_dir=tmp_path / "output",
        )
    )

    assert config.model.load_bf16 is True


def test_training_config_does_not_expose_fixed_method_config():
    from examples.RoboTwin2.train import RoboTwin2TrainConfig

    public_fields = {field.name for field in fields(RoboTwin2TrainConfig)}
    assert "modality_config_path" not in public_fields
    assert "experiment_name" not in public_fields
    assert "max_steps" not in public_fields
    assert "save_steps" not in public_fields


@pytest.mark.parametrize(
    ("stage", "expected_max_steps", "expected_save_steps"),
    (
        ("warmup", 20_000, 20_000),
        ("full", 120_000, 40_000),
    ),
)
def test_build_config_fixes_stage_schedule_and_public_output_dir(
    tmp_path: Path,
    stage: str,
    expected_max_steps: int,
    expected_save_steps: int,
):
    from examples.RoboTwin2.train import RoboTwin2TrainConfig, build_config

    public_output_dir = tmp_path / f"{stage}-output"
    config = build_config(
        RoboTwin2TrainConfig(
            stage=stage,
            base_model_path="base-model",
            dataset_path=tmp_path / "dataset",
            output_dir=public_output_dir,
        )
    )
    assert Path(config.training.output_dir) == public_output_dir
    assert config.training.experiment_name is None
    assert config.training.max_steps == expected_max_steps
    assert config.training.save_steps == expected_save_steps


def test_full_experiment_uses_warmup_weights_with_an_independent_schedule_and_output(
    monkeypatch,
    tmp_path: Path,
):
    from examples.RoboTwin2.train import RoboTwin2TrainConfig, build_config
    from gr00t.experiment import experiment

    warmup_checkpoint = tmp_path / "warmup" / "checkpoint-20000"
    warmup_checkpoint.mkdir(parents=True)
    output_dir = tmp_path / "full"
    config = build_config(
        RoboTwin2TrainConfig(
            stage="full",
            base_model_path=str(warmup_checkpoint),
            dataset_path=tmp_path / "dataset",
            output_dir=output_dir,
            num_gpus=1,
        )
    )
    captured = {}

    class FakeProcessor:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    class FakePipeline:
        def __init__(self, runtime_config, save_cfg_dir):
            captured["pipeline_output"] = Path(save_cfg_dir).parent
            captured["model_path"] = runtime_config.training.start_from_checkpoint
            self.model = torch.nn.Linear(1, 1)
            self.processor = FakeProcessor()

        def setup(self):
            pass

        def return_model(self):
            return self.model

        def return_dataset(self):
            return object(), None

        def return_collator(self):
            return lambda values: values

        def return_processor(self):
            return self.processor

    class FakeTrainer:
        def __init__(self, *, args, **kwargs):
            captured["trainer_output"] = Path(args.output_dir)
            captured["max_steps"] = args.max_steps
            captured["save_steps"] = args.save_steps
            self.loss = torch.zeros(())

        def add_callback(self, callback):
            pass

        def train(self, resume_from_checkpoint=None):
            captured["resume_from_checkpoint"] = resume_from_checkpoint

        def save_model(self):
            pass

    monkeypatch.setitem(experiment.MODEL_REGISTRY, type(config.model), FakePipeline)
    monkeypatch.setattr(experiment, "Gr00tTrainer", FakeTrainer)

    experiment.run(config)

    assert captured == {
        "pipeline_output": output_dir,
        "model_path": str(warmup_checkpoint),
        "trainer_output": output_dir,
        "max_steps": 120_000,
        "save_steps": 40_000,
        "resume_from_checkpoint": True,
    }


def test_public_shell_surface_contains_only_the_reproducible_pipeline():
    expected = {
        "train_query_warmup.sh",
        "train_c2_full.sh",
        "eval_server.sh",
        "eval_client.sh",
    }
    assert {path.name for path in EXAMPLE_DIR.glob("*.sh")} == expected
    assert {path.name for path in EXAMPLE_DIR.glob("*.py")} == {
        "modality_config_robotwin2.py",
        "train.py",
        "validate_dataset.py",
    }
    assert (EXAMPLE_DIR / "README.md").is_file()


def test_training_launchers_use_the_fixed_entrypoint_without_passthrough():
    for name in ("train_query_warmup.sh", "train_c2_full.sh"):
        text = (EXAMPLE_DIR / name).read_text()
        assert "examples/RoboTwin2/train.py" in text
        assert '"$@"' not in text


@pytest.mark.parametrize("script_name", ("train_query_warmup.sh", "train_c2_full.sh"))
def test_training_launcher_arguments_are_accepted_by_real_tyro_parser(
    tmp_path: Path,
    script_name: str,
):
    import tyro

    from examples.RoboTwin2.train import RoboTwin2TrainConfig

    base_model = tmp_path / "base-model"
    dataset = tmp_path / "dataset"
    warmup = tmp_path / "warmup"
    base_model.mkdir()
    dataset.mkdir()
    (warmup / "checkpoint-20000").mkdir(parents=True)
    environment = {
        **os.environ,
        "BASE_MODEL_PATH": str(base_model),
        "DATASET_PATH": str(dataset),
        "WARMUP_OUTPUT_DIR": str(warmup),
        "NUM_GPUS": "1",
        "GPU_IDS": "0",
        "PYTHON": "/bin/echo",
    }

    result = subprocess.run(
        ["bash", str(EXAMPLE_DIR / script_name)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    launcher_argv = shlex.split(result.stdout.splitlines()[-1])
    train_entrypoint = str(EXAMPLE_DIR / "train.py")
    train_args = launcher_argv[launcher_argv.index(train_entrypoint) + 1 :]
    parsed = tyro.cli(RoboTwin2TrainConfig, args=train_args)

    assert parsed.stage == ("warmup" if "warmup" in script_name else "full")


def test_training_launchers_expose_only_operational_overrides():
    for name in ("train_query_warmup.sh", "train_c2_full.sh"):
        text = (EXAMPLE_DIR / name).read_text()
        for variable, option in (
            ("GLOBAL_BATCH_SIZE", "--global-batch-size"),
            ("LEARNING_RATE", "--learning-rate"),
            ("DATALOADER_NUM_WORKERS", "--dataloader-num-workers"),
        ):
            assert variable in text
            assert option in text
        assert "MAX_STEPS=" not in text
        assert "SAVE_STEPS=" not in text

    warmup = (EXAMPLE_DIR / "train_query_warmup.sh").read_text()
    full = (EXAMPLE_DIR / "train_c2_full.sh").read_text()
    assert "--max-steps" not in warmup
    assert "--save-steps" not in warmup
    assert "--max-steps" not in full
    assert "--save-steps" not in full


def test_full_launcher_defaults_to_the_warmup_20k_checkpoint():
    warmup_text = (EXAMPLE_DIR / "train_query_warmup.sh").read_text()
    text = (EXAMPLE_DIR / "train_c2_full.sh").read_text()

    assert 'WARMUP_OUTPUT_DIR="${WARMUP_OUTPUT_DIR:-' in warmup_text
    assert '--output-dir "${WARMUP_OUTPUT_DIR}"' in warmup_text
    assert 'WARMUP_OUTPUT_DIR="${WARMUP_OUTPUT_DIR:-' in text
    assert 'WARMUP_CHECKPOINT="${WARMUP_OUTPUT_DIR}/checkpoint-20000"' in text
    assert '--base-model-path "${WARMUP_CHECKPOINT}"' in text
    assert 'FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-' in text
    assert '--output-dir "${FULL_OUTPUT_DIR}"' in text


def test_server_defaults_to_the_full_120k_checkpoint():
    text = (EXAMPLE_DIR / "eval_server.sh").read_text()

    assert 'FULL_OUTPUT_DIR="${FULL_OUTPUT_DIR:-' in text
    assert 'CHECKPOINT="${CHECKPOINT:-${FULL_OUTPUT_DIR}/checkpoint-120000}"' in text


def test_eval_client_fixes_the_public_evaluation_method():
    text = (EXAMPLE_DIR / "eval_client.sh").read_text()

    assert "CHUNK=" not in text
    assert "TASK_CONFIG=" not in text
    assert "--chunk_exec full" in text
    assert "--task_config demo_clean" in text


@pytest.mark.parametrize("script_name", ("train_query_warmup.sh", "train_c2_full.sh"))
def test_training_launcher_rejects_gpu_count_mismatch(
    tmp_path: Path,
    script_name: str,
):
    base_model = tmp_path / "base-model"
    dataset = tmp_path / "dataset"
    warmup = tmp_path / "warmup"
    base_model.mkdir()
    dataset.mkdir()
    (warmup / "checkpoint-20000").mkdir(parents=True)
    environment = {
        **os.environ,
        "BASE_MODEL_PATH": str(base_model),
        "DATASET_PATH": str(dataset),
        "WARMUP_OUTPUT_DIR": str(warmup),
        "NUM_GPUS": "2",
        "GPU_IDS": "0,1,2",
        "PYTHON": "/bin/false",
    }

    result = subprocess.run(
        ["bash", str(EXAMPLE_DIR / script_name)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "NUM_GPUS=2" in result.stderr
    assert "GPU_IDS contains 3 IDs" in result.stderr


@pytest.mark.parametrize("script_name", ("train_query_warmup.sh", "train_c2_full.sh"))
def test_training_launcher_rejects_nonpositive_gpu_count(
    tmp_path: Path,
    script_name: str,
):
    environment = {
        **os.environ,
        "BASE_MODEL_PATH": str(tmp_path),
        "DATASET_PATH": str(tmp_path),
        "WARMUP_OUTPUT_DIR": str(tmp_path),
        "NUM_GPUS": "0",
        "GPU_IDS": "0",
        "PYTHON": "/bin/false",
    }

    result = subprocess.run(
        ["bash", str(EXAMPLE_DIR / script_name)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "NUM_GPUS must be a positive integer" in result.stderr


@pytest.mark.parametrize("script_name", ("train_query_warmup.sh", "train_c2_full.sh"))
@pytest.mark.parametrize(
    ("gpu_ids", "error_text"),
    (
        ("0,0", "GPU_IDS must contain unique IDs"),
        ("0,00", "GPU_IDS must contain unique IDs"),
        ("0,gpu1", "GPU_IDS must be comma-separated non-negative integer IDs"),
        ("0,", "GPU_IDS must be comma-separated non-negative integer IDs"),
        (",0", "GPU_IDS must be comma-separated non-negative integer IDs"),
        ("0,,1", "GPU_IDS must be comma-separated non-negative integer IDs"),
    ),
)
def test_training_launcher_rejects_invalid_gpu_ids(
    tmp_path: Path,
    script_name: str,
    gpu_ids: str,
    error_text: str,
):
    base_model = tmp_path / "base-model"
    dataset = tmp_path / "dataset"
    warmup = tmp_path / "warmup"
    base_model.mkdir()
    dataset.mkdir()
    (warmup / "checkpoint-20000").mkdir(parents=True)
    environment = {
        **os.environ,
        "BASE_MODEL_PATH": str(base_model),
        "DATASET_PATH": str(dataset),
        "WARMUP_OUTPUT_DIR": str(warmup),
        "FULL_OUTPUT_DIR": str(tmp_path / "full"),
        "NUM_GPUS": "2",
        "GPU_IDS": gpu_ids,
        "PYTHON": "/bin/true",
    }

    result = subprocess.run(
        ["bash", str(EXAMPLE_DIR / script_name)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert error_text in result.stderr


@pytest.mark.parametrize("alias_kind", ("same-path", "symlink"))
def test_full_launcher_rejects_warmup_output_alias_as_full_output(
    tmp_path: Path,
    alias_kind: str,
):
    dataset = tmp_path / "dataset"
    warmup = tmp_path / "warmup"
    dataset.mkdir()
    (warmup / "checkpoint-20000").mkdir(parents=True)

    if alias_kind == "same-path":
        full_output = warmup
    else:
        full_output = tmp_path / "full-alias"
        full_output.symlink_to(warmup, target_is_directory=True)

    result = subprocess.run(
        ["bash", str(EXAMPLE_DIR / "train_c2_full.sh")],
        env={
            **os.environ,
            "DATASET_PATH": str(dataset),
            "WARMUP_OUTPUT_DIR": str(warmup),
            "FULL_OUTPUT_DIR": str(full_output),
            "NUM_GPUS": "1",
            "GPU_IDS": "0",
            # The no-op executable proves validation stops before either Python call.
            "PYTHON": "/bin/true",
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "FULL_OUTPUT_DIR must be different from WARMUP_OUTPUT_DIR" in result.stderr
