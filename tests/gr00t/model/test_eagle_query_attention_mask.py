"""Eagle attention contracts for the fixed 75+75 query groups."""

import torch

from gr00t.model.modules.eagle_backbone import (
    _build_disentangled_query_attention_mask,
)


def test_disentangled_query_attention_blocks_cross_query_attention() -> None:
    base_len = 3
    action_query_len = 2
    segmentation_query_len = 2
    attention_mask = torch.ones(1, base_len + action_query_len + segmentation_query_len)

    mask = _build_disentangled_query_attention_mask(
        attention_mask=attention_mask,
        base_seq_len=base_len,
        action_query_num_tokens=action_query_len,
        segmentation_query_num_tokens=segmentation_query_len,
        dtype=torch.float32,
    )

    allowed = mask[0, 0] == 0
    action_start = base_len
    segmentation_start = base_len + action_query_len
    assert allowed[action_start, action_start + 1]
    assert not allowed[action_start, segmentation_start]
    assert not allowed[segmentation_start, action_start]
    assert allowed[segmentation_start, segmentation_start + 1]


def test_disentangled_query_attention_keeps_padding_blocked() -> None:
    attention_mask = torch.tensor([[1, 0, 1, 1, 1]], dtype=torch.long)

    mask = _build_disentangled_query_attention_mask(
        attention_mask=attention_mask,
        base_seq_len=3,
        action_query_num_tokens=1,
        segmentation_query_num_tokens=1,
        dtype=torch.float32,
    )

    assert (mask[0, 0, :, 1] < 0).all()


def test_disentangled_query_attention_covers_two_fixed_75_token_groups() -> None:
    base_len = 7
    query_len = 3 * 5 * 5
    total_len = base_len + 2 * query_len
    attention_mask = torch.ones(1, total_len)

    mask = _build_disentangled_query_attention_mask(
        attention_mask=attention_mask,
        base_seq_len=base_len,
        action_query_num_tokens=query_len,
        segmentation_query_num_tokens=query_len,
        dtype=torch.float32,
    )

    allowed = mask[0, 0] == 0
    action_start = base_len
    segmentation_start = base_len + query_len
    assert mask.shape == (1, 1, total_len, total_len)
    assert allowed[action_start, action_start + query_len - 1]
    assert allowed[segmentation_start, segmentation_start + query_len - 1]
    assert not allowed[action_start, segmentation_start]
    assert not allowed[segmentation_start, action_start]
