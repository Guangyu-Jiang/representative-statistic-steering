import json

import pytest
import torch

from repstat_steering.pplm_control import resolve_margin_target
from repstat_steering.pplm_quantiles import load_quantile_margin_targets


def test_direct_margin_target_is_used_without_current_state_shift() -> None:
    current = torch.tensor(-2.0)
    target = resolve_margin_target(
        current,
        target_probability=0.8,
        target_margin=1.25,
    )
    assert torch.isclose(target, torch.tensor(1.25))


def test_probability_floor_can_raise_quantile_margin() -> None:
    current = torch.tensor(-2.0)
    target = resolve_margin_target(
        current,
        target_probability=0.8,
        target_margin=-1.0,
        minimum_target_probability=0.75,
    )
    assert torch.isclose(target, torch.logit(torch.tensor(0.75)))


def test_direct_and_relative_margin_targets_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_margin_target(
            torch.tensor(0.0),
            target_probability=0.8,
            target_margin=1.0,
            target_margin_shift=0.5,
        )


def test_load_class_specific_quantile_targets(tmp_path) -> None:
    calibration = tmp_path / "targets.json"
    calibration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "targets": {
                    "positive": {"quantiles": {"0.5": 1.0, "0.9": 3.0}},
                    "negative": {"quantiles": {"0.5": 2.0, "0.9": 4.0}},
                },
            }
        ),
        encoding="utf-8",
    )
    assert load_quantile_margin_targets(calibration, 0.9) == {
        "positive": 3.0,
        "negative": 4.0,
    }
