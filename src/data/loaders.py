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


def synthetic_clients(seed: int, num_clients: int, days: int = 30, steps_per_day: int = 24) -> dict[str, np.ndarray]:
    """Hourly load for several homes that differ in level, peak size and peak time (non-IID).

    Use only to test the pipeline. These are not real measurements.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(days * steps_per_day)
    clients = {}
    for index in range(num_clients):
        base = rng.uniform(0.5, 2.0)
        amplitude = rng.uniform(0.2, 0.8) * base
        peak_shift = rng.uniform(0, steps_per_day)
        noise = rng.normal(0, 0.05 * base, t.size)
        clients[f"client_{index:02d}"] = base + amplitude * np.sin(2 * np.pi * (t - peak_shift) / steps_per_day) + noise
    return clients


def load_clients(data_cfg: dict, seed: int, num_clients: int) -> dict[str, np.ndarray]:
    """Return one series per client, as described by the `data` section of a config.

    - `synthetic_clients`: `num_clients` generated homes.
    - `csv`: one file per client, listed under `paths`. The client name is the file name.
    """
    source = data_cfg["source"]
    if source == "synthetic_clients":
        return synthetic_clients(seed, num_clients)
    if source == "csv":
        paths = [Path(p) for p in data_cfg.get("paths") or []]
        if not paths:
            raise ValueError("Federated runs on csv data need `data.paths`: one file per client")
        return {p.stem: pd.read_csv(p)[data_cfg["target_column"]].to_numpy(dtype=float) for p in paths}
    raise ValueError(f"Unknown data source for federated runs: {source}")
