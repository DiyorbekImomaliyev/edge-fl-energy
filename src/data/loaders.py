"""Load one energy time series and split it in time order."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_series(data_cfg: dict, seed: int) -> np.ndarray:
    """Return the target series described by the `data` section of a config."""
    source = data_cfg["source"]
    if source == "synthetic":
        return synthetic_load(seed)
    if source == "csv":
        frame = pd.read_csv(Path(data_cfg["path"]))
        return frame[data_cfg["target_column"]].to_numpy(dtype=float)
    raise ValueError(f"Unknown data source: {source}")


def synthetic_load(seed: int, days: int = 30, steps_per_day: int = 96) -> np.ndarray:
    """Daily load cycle with noise at 15-minute steps. Use only to test the pipeline."""
    rng = np.random.default_rng(seed)
    t = np.arange(days * steps_per_day)
    return 1.0 + 0.5 * np.sin(2 * np.pi * t / steps_per_day) + rng.normal(0, 0.05, t.size)


def train_test_split(series: np.ndarray, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep time order: the last `test_fraction` of the series is the test set."""
    cut = int(len(series) * (1 - test_fraction))
    return series[:cut], series[cut:]
