from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import Trainer

from gr00t.experiment.trainer import Gr00tTrainer


@pytest.mark.parametrize("stage", ("warmup", "full"))
def test_trainer_logs_only_selected_loss_components(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    trainer = object.__new__(Gr00tTrainer)
    trainer.action_offset = None
    trainer._aux_log_sums = {}
    trainer._aux_log_count = 0
    trainer.state = SimpleNamespace(epoch=1.0, global_step=1)
    trainer.args = SimpleNamespace(logging_steps=10, local_rank=-1)
    trainer._nested_gather = lambda value: value

    total_offset = 0.0 if stage == "warmup" else 1.0
    batches = iter(
        (
            (
                torch.tensor(1.0 + total_offset),
                {
                    "action_loss_scalar": torch.tensor(2.0),
                    "depth_loss": torch.tensor(3.0),
                    "segmentation_loss": torch.tensor(4.0),
                    "depth_loss_terms": {"mae": torch.tensor(30.0)},
                    "segmentation_loss_terms": {
                        "cross_entropy": torch.tensor(40.0)
                    },
                },
            ),
            (
                torch.tensor(3.0 + total_offset),
                {
                    "action_loss_scalar": torch.tensor(4.0),
                    "depth_loss": torch.tensor(5.0),
                    "segmentation_loss": torch.tensor(6.0),
                },
            ),
        )
    )

    def fake_compute_loss(*args, **kwargs):
        del args, kwargs
        return next(batches)

    logged = []
    monkeypatch.setattr(Trainer, "compute_loss", fake_compute_loss)
    monkeypatch.setattr(
        Trainer,
        "log",
        lambda self, logs, start_time=None: logged.append(dict(logs)),
    )

    model = torch.nn.Linear(1, 1)
    model.train()
    trainer.compute_loss(model, {})
    trainer.compute_loss(model, {})
    trainer.log({"loss": 2.0 + total_offset})

    assert logged == [
        {
            "loss": 2.0 + total_offset,
            "total/loss": 2.0 + total_offset,
            "action/loss": 3.0,
            "depth/loss": 4.0,
            "segmentation/loss": 5.0,
        }
    ]
    assert trainer._aux_log_sums == {}
    assert trainer._aux_log_count == 0
