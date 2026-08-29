"""Exact-name loading for base, Warmup, and Full V-Link checkpoints."""

import json
import logging
import os
from pathlib import Path

import torch


def load_checkpoint_shard(path: Path) -> dict[str, torch.Tensor]:
    """Load a PyTorch or safetensors checkpoint shard on CPU."""
    suffix = path.suffix
    if suffix == ".safetensors":
        try:
            from transformers.modeling_utils import load_state_dict

            return load_state_dict(str(path))
        except Exception:
            from safetensors.torch import load_file

            return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu")


def checkpoint_shard_paths(checkpoint_path: Path) -> list[Path]:
    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        return [checkpoint_path / shard_name for shard_name in shard_names]

    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = checkpoint_path / filename
        if path.exists():
            return [path]

    raise FileNotFoundError(
        f"No model.safetensors(.index.json) or pytorch_model.bin found in {checkpoint_path}"
    )


def _key_group(key: str) -> str:
    if "action_query_bridge" in key:
        return "action_query_bridge"
    if key.startswith("backbone.action_queries") or key.startswith(
        "backbone.action_query_spatial_pos_embed"
    ):
        return "action_query_tokens"
    if key.startswith("backbone.segmentation_queries") or key.startswith(
        "backbone.segmentation_query_spatial_pos_embed"
    ):
        return "segmentation_query_tokens"
    if "action_query_depth_head" in key:
        return "action_query_depth_head"
    if "action_query_segmentation_head" in key:
        return "action_query_segmentation_head"
    if is_vlm_lora_key(key):
        return "vlm_lora"
    if key.startswith("backbone."):
        return "backbone"
    if key.startswith("action_head.model."):
        return "action_dit"
    if key.startswith("action_head."):
        return "action_head_other"
    return key.split(".", 1)[0]


def _group_counts(keys: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key in keys:
        group = _key_group(key)
        counts[group] = counts.get(group, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _log_key_list(title: str, keys: list[str], *, level: int = logging.WARNING) -> None:
    if not keys:
        logging.log(level, "%s: none", title)
        return

    limit = int(os.environ.get("GR00T_CHECKPOINT_KEY_LOG_LIMIT", "200"))
    logging.log(level, "%s: count=%d groups=%s", title, len(keys), _group_counts(keys))
    for key in keys[:limit]:
        logging.log(level, "  - %s", key)
    if len(keys) > limit:
        logging.log(
            level,
            "  ... %d more keys omitted; set GR00T_CHECKPOINT_KEY_LOG_LIMIT to show more",
            len(keys) - limit,
        )


def _log_mismatched_key_list(
    title: str,
    items: list[tuple[str, tuple[int, ...], tuple[int, ...]]],
) -> None:
    if not items:
        logging.warning("%s: none", title)
        return

    limit = int(os.environ.get("GR00T_CHECKPOINT_KEY_LOG_LIMIT", "200"))
    keys = [item[0] for item in items]
    logging.warning("%s: count=%d groups=%s", title, len(items), _group_counts(keys))
    for name, checkpoint_shape, model_shape in items[:limit]:
        logging.warning(
            "  - %s checkpoint_shape=%s current_model_shape=%s",
            name,
            checkpoint_shape,
            model_shape,
        )
    if len(items) > limit:
        logging.warning(
            "  ... %d more mismatched keys omitted; set GR00T_CHECKPOINT_KEY_LOG_LIMIT to show more",
            len(items) - limit,
        )


def load_pretrained_weights_by_name(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    strict_lora_shape: bool = False,
) -> dict[str, list]:
    """Load checkpoint tensors only when name and shape match the target model."""
    model_state = model.state_dict()
    loaded_keys: set[str] = set()
    unexpected_keys: list[str] = []
    mismatched_keys: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for shard_path in checkpoint_shard_paths(checkpoint_path):
        shard_state = load_checkpoint_shard(shard_path)
        for name, tensor in shard_state.items():
            if name not in model_state:
                unexpected_keys.append(name)
                continue
            target = model_state[name]
            if tuple(target.shape) != tuple(tensor.shape):
                mismatched_keys.append((name, tuple(tensor.shape), tuple(target.shape)))
                continue
            with torch.no_grad():
                target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded_keys.add(name)
        del shard_state

    missing_keys = [name for name in model_state.keys() if name not in loaded_keys]
    logging.info(
        "Loaded checkpoint by exact name/shape from %s: loaded=%d missing=%d "
        "unexpected=%d mismatched=%d",
        checkpoint_path,
        len(loaded_keys),
        len(missing_keys),
        len(unexpected_keys),
        len(mismatched_keys),
    )
    if mismatched_keys:
        _log_mismatched_key_list(
            "HIGHLIGHT checkpoint params NOT loaded due to shape mismatch",
            mismatched_keys,
        )
        if strict_lora_shape:
            mismatched_lora_keys = [
                item for item in mismatched_keys if is_vlm_lora_key(item[0])
            ]
            if mismatched_lora_keys:
                raise ValueError(
                    "VLM LoRA checkpoint keys exist but their shapes do not match the "
                    "current model. This usually means --vlm_lora_rank differs from "
                    f"the checkpoint rank. First mismatched LoRA keys: {mismatched_lora_keys[:20]}"
                )
    else:
        logging.info("HIGHLIGHT checkpoint params NOT loaded due to shape mismatch: none")
    if unexpected_keys:
        _log_key_list(
            "HIGHLIGHT checkpoint params NOT loaded because key is absent in current model",
            unexpected_keys,
        )
    else:
        logging.info(
            "HIGHLIGHT checkpoint params NOT loaded because key is absent in current model: none"
        )
    if missing_keys:
        _log_key_list(
            "HIGHLIGHT current model params NOT initialized from checkpoint",
            missing_keys,
            level=logging.WARNING,
        )
    else:
        logging.info("HIGHLIGHT current model params NOT initialized from checkpoint: none")

    return {
        "loaded_keys": sorted(loaded_keys),
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "mismatched_keys": mismatched_keys,
    }


def _group_model_keys(model_state_keys: list[str], group_name: str) -> list[str]:
    return sorted(key for key in model_state_keys if _key_group(key) == group_name)


def _group_checkpoint_keys(loading_info: dict[str, list], group_name: str) -> list[str]:
    loaded_keys = [
        key for key in loading_info.get("loaded_keys", []) if _key_group(key) == group_name
    ]
    unexpected_keys = [
        key for key in loading_info.get("unexpected_keys", []) if _key_group(key) == group_name
    ]
    mismatched_keys = [
        item[0]
        for item in loading_info.get("mismatched_keys", [])
        if _key_group(item[0]) == group_name
    ]
    return sorted(set(loaded_keys + unexpected_keys + mismatched_keys))


def _shape_by_key(model: torch.nn.Module, keys: list[str]) -> dict[str, tuple[int, ...]]:
    model_state = model.state_dict()
    return {key: tuple(model_state[key].shape) for key in keys if key in model_state}


def confirm_checkpoint_group_loading(
    model: torch.nn.Module,
    loading_info: dict[str, list],
    *,
    required_groups: tuple[str, ...] = (),
    must_load_groups: tuple[str, ...] = (),
    allow_unexpected_groups: tuple[str, ...] = (),
) -> dict[str, dict]:
    """Confirm critical checkpoint groups loaded by exact key and shape.

    This catches the dangerous case where a checkpoint contains a trained group
    such as Segmentation Query tokens, but the current model shape differs, so
    exact-name loading silently skips the group and leaves it reinitialized.
    """
    model_state_keys = list(model.state_dict().keys())
    loaded_keys = set(loading_info.get("loaded_keys", []))
    missing_keys = set(loading_info.get("missing_keys", []))
    mismatched_by_key = {
        item[0]: (tuple(item[1]), tuple(item[2]))
        for item in loading_info.get("mismatched_keys", [])
    }
    summaries: dict[str, dict] = {}

    groups_to_check = tuple(dict.fromkeys((*required_groups, *must_load_groups)))
    must_load_group_set = set(must_load_groups)
    allow_unexpected_group_set = set(allow_unexpected_groups)

    for group_name in groups_to_check:
        label = group_name.upper()
        model_keys = _group_model_keys(model_state_keys, group_name)
        checkpoint_keys = _group_checkpoint_keys(loading_info, group_name)
        loaded_group_keys = sorted(key for key in model_keys if key in loaded_keys)
        missing_group_keys = sorted(key for key in model_keys if key in missing_keys)
        mismatched_group_keys = {
            key: shapes
            for key, shapes in mismatched_by_key.items()
            if _key_group(key) == group_name
        }
        unexpected_group_keys = sorted(
            key
            for key in loading_info.get("unexpected_keys", [])
            if _key_group(key) == group_name
        )

        summary = {
            "status": "inactive",
            "model_key_count": len(model_keys),
            "checkpoint_key_count": len(checkpoint_keys),
            "loaded_key_count": len(loaded_group_keys),
            "model_shapes": _shape_by_key(model, model_keys),
            "mismatched_keys": mismatched_group_keys,
            "unexpected_keys": unexpected_group_keys,
        }
        summaries[group_name] = summary

        if not model_keys:
            logging.info("CHECKPOINT LOAD CONFIRMATION %s: inactive in current model", label)
            continue

        if not checkpoint_keys:
            summary["status"] = "absent_in_checkpoint"
            if group_name in must_load_group_set:
                raise ValueError(
                    f"CHECKPOINT LOAD CONFIRMATION {label} failed: current model "
                    f"requires {group_name} to be loaded, but the checkpoint contains "
                    "no matching keys for that group. Check the base checkpoint path "
                    "and query/LoRA configuration."
                )
            logging.warning(
                "CHECKPOINT LOAD CONFIRMATION %s: absent from checkpoint; current model "
                "parameters will remain initialized. model_shapes=%s",
                label,
                summary["model_shapes"],
            )
            continue

        if (
            group_name in must_load_group_set
            and group_name not in allow_unexpected_group_set
            and unexpected_group_keys
        ):
            summary["status"] = "failed"
            raise ValueError(
                f"CHECKPOINT LOAD CONFIRMATION {label} failed: active checkpoint "
                f"contains unexpected {group_name} keys that do not exist in the "
                f"current model: {unexpected_group_keys}. The source and target "
                "parameter schemas must match exactly."
            )

        if mismatched_group_keys or missing_group_keys:
            summary["status"] = "failed"
            raise ValueError(
                f"CHECKPOINT LOAD CONFIRMATION {label} failed: checkpoint contains "
                f"{group_name}, but not all current model parameters were loaded by exact "
                f"name/shape. loaded={loaded_group_keys} missing={missing_group_keys} "
                f"mismatched={mismatched_group_keys}. This usually means the current "
                "token count/grid/rank does not match the checkpoint."
            )

        summary["status"] = "loaded"
        logging.info(
            "CHECKPOINT LOAD CONFIRMATION %s: loaded %d/%d keys. model_shapes=%s",
            label,
            len(loaded_group_keys),
            len(model_keys),
            summary["model_shapes"],
        )

    return summaries


def is_vlm_lora_key(key: str) -> bool:
    return "backbone.model.language_model" in key and "lora_" in key


def checkpoint_has_vlm_lora(checkpoint_path: Path) -> bool:
    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        return any(is_vlm_lora_key(key) for key in index["weight_map"].keys())

    for shard_path in checkpoint_shard_paths(checkpoint_path):
        shard_state = load_checkpoint_shard(shard_path)
        try:
            if any(is_vlm_lora_key(key) for key in shard_state.keys()):
                return True
        finally:
            del shard_state
    return False
