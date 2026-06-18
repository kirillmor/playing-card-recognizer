from __future__ import annotations

from card_recognizer.training.mlflow_utils import flatten_dict


def test_flatten_dict() -> None:
    flattened = flatten_dict(
        {
            "model": {
                "name": "baseline_cnn",
                "channels": [32, 64],
            },
            "seed": 42,
        }
    )

    assert flattened == {
        "model.name": "baseline_cnn",
        "model.channels": "[32, 64]",
        "seed": 42,
    }
