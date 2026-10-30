from pathlib import Path

import numpy as np
import pytest
import yaml

from src.evaluation.metrics import mae, rmse
from src.pipeline import run

ROOT = Path(__file__).resolve().parents[1]


def test_example_config_runs():
    config = yaml.safe_load((ROOT / "configs/EXP-000_persistence_baseline.yaml").read_text())
    metrics = run(config)
    assert metrics["test_size"] > 0
    assert 0 <= metrics["mae"] <= metrics["rmse"]


def test_metrics_on_known_values():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    assert mae(y_true, y_pred) == pytest.approx(2 / 3)
    assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(4 / 3))
