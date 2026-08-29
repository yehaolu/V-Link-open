"""Behavior contracts for the fixed RoboTwin2 public inference policy."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor

from examples.RoboTwin2.modality_config_robotwin2 import robotwin2_config
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.eval import run_gr00t_server
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper


class _FakeModel:
    def eval(self) -> None:
        self.did_eval = True

    def to(self, **kwargs) -> None:
        self.to_kwargs = kwargs

    def get_action(self, **kwargs) -> dict[str, torch.Tensor]:
        return {"action_pred": torch.zeros(1, 15, 128)}


class _FakeProcessor:
    def __init__(self) -> None:
        self.seen_steps = []
        self.collator = lambda processed: {}

    def eval(self) -> None:
        pass

    def get_modality_configs(self) -> dict:
        return {
            EmbodimentTag.NEW_EMBODIMENT.value: robotwin2_config
        }

    def __call__(self, messages: list[dict]) -> dict:
        self.seen_steps.append(messages[0]["content"])
        return {}

    def decode_action(self, action, embodiment_tag, batched_states) -> dict:
        return {
            key: np.zeros((1, 15, 1), dtype=np.float32)
            for key in robotwin2_config["action"].modality_keys
        }


class _CapturingServer:
    instance = None

    def __init__(self, policy, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = port
        self.did_run = False
        type(self).instance = self

    def run(self) -> None:
        self.did_run = True


def _flat_inference_observation() -> dict[str, object]:
    observation = {
        f"video.{key}": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)
        for key in robotwin2_config["video"].modality_keys
    }
    observation.update(
        {
            f"state.{key}": np.zeros((1, 1, 1), dtype=np.float32)
            for key in robotwin2_config["state"].modality_keys
        }
    )
    language_key = robotwin2_config["language"].modality_keys[0]
    observation[language_key] = ["pick up the object"]

    # Deliberately supplied as unadvertised extras. The public action path must
    # neither validate nor forward this training-only target transport.
    observation.update(
        {
            "camera.K": np.zeros((1, 1, 3, 3, 3), dtype=np.float32),
            "camera.cam2world": np.zeros((1, 1, 3, 4, 4), dtype=np.float32),
            "camera.depth_m": np.zeros((1, 1, 3, 4, 4), dtype=np.float32),
            "camera.mesh_segmentation": np.zeros((1, 1, 3, 4, 4), dtype=np.int64),
        }
    )
    return observation


def test_eval_server_uses_only_exact_name_loader_and_public_inference_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    model = _FakeModel()
    processor = _FakeProcessor()
    exact_name_loader = Mock(return_value=model)
    fallback_loader = Mock(side_effect=AssertionError("AutoModel fallback was called"))
    processor_loader = Mock(return_value=processor)

    monkeypatch.setattr(Gr00tN1d6, "from_pretrained_by_key", exact_name_loader)
    monkeypatch.setattr(AutoModel, "from_pretrained", fallback_loader)
    monkeypatch.setattr(AutoProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(run_gr00t_server, "PolicyServer", _CapturingServer)

    run_gr00t_server.main(
        run_gr00t_server.ServerConfig(
            model_path=str(tmp_path),
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            device="cpu",
            host="127.0.0.1",
            port=5710,
            use_sim_policy_wrapper=True,
        )
    )

    exact_name_loader.assert_called_once_with(
        tmp_path,
        transformers_loading_kwargs={"trust_remote_code": True},
    )
    fallback_loader.assert_not_called()
    processor_loader.assert_called_once_with(tmp_path)
    assert model.did_eval
    assert model.to_kwargs == {"device": "cpu", "dtype": torch.bfloat16}

    server = _CapturingServer.instance
    assert server is not None and server.did_run
    assert isinstance(server.policy, Gr00tSimPolicyWrapper)
    assert set(server.policy.get_modality_config()) == {
        "video",
        "state",
        "action",
        "language",
    }

    action, info = server.policy.get_action(_flat_inference_observation())
    assert set(action) == {
        f"action.{key}"
        for key in robotwin2_config["action"].modality_keys
    }
    assert info == {}
    assert len(processor.seen_steps) == 1
    assert processor.seen_steps[0].camera_calibration is None


def test_robotwin_modality_executes_depth_and_segmentation_target_transport() -> None:
    camera_modality = robotwin2_config["camera_calibration"]
    assert camera_modality.delta_indices == [0]
    assert camera_modality.modality_keys == ["camera_calibration"]

    row = {}
    for modality, config in robotwin2_config.items():
        if modality == "camera_calibration":
            continue
        for key in config.modality_keys:
            if modality in {"state", "action"}:
                value = np.zeros(1, dtype=np.float32)
            elif modality == "language":
                value = "pick up the object"
            else:
                value = np.zeros((4, 4, 3), dtype=np.uint8)
            row[f"{modality}.{key}"] = [value]
    episode = pd.DataFrame(row)

    depth_m = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
    mesh_segmentation = np.arange(12, dtype=np.int64).reshape(1, 3, 2, 2)
    episode_calibration = {
        "K": np.broadcast_to(np.eye(3, dtype=np.float32), (1, 3, 3, 3)).copy(),
        "cam2world": np.broadcast_to(
            np.eye(4, dtype=np.float32), (1, 3, 4, 4)
        ).copy(),
        "depth_m": depth_m,
        "mesh_segmentation": mesh_segmentation,
        "view_names": ["image", "left_wrist_image", "right_wrist_image"],
    }

    step = extract_step_data(
        episode,
        step_index=0,
        modality_configs=robotwin2_config,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        allow_padding=True,
        episode_calibration=episode_calibration,
    )

    assert step.camera_calibration is not None
    np.testing.assert_array_equal(step.camera_calibration["depth_m"], depth_m)
    np.testing.assert_array_equal(
        step.camera_calibration["mesh_segmentation"], mesh_segmentation
    )
