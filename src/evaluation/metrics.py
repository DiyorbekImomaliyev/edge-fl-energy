"""Error metrics for forecasts."""
import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent. Points where the true value is zero are left out."""
    keep = y_true != 0
    if not keep.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[keep] - y_pred[keep]) / y_true[keep])) * 100)
