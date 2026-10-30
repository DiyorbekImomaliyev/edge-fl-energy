"""Baselines that every new model must beat."""
import numpy as np


def persistence_forecast(train: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Predict each test value as the value one step before it."""
    return np.concatenate([train[-1:], test[:-1]])
