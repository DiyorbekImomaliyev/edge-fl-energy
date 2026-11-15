"""Turn a load series into (input window, target) pairs for forecasting models."""
import numpy as np


def make_windows(series: np.ndarray, lookback: int, horizon: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inputs, targets and the position of each target in `series`.

    Each input is `lookback` consecutive values. Its target is the value
    `horizon` steps after the last input value (horizon=1 is the next step).
    """
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be at least 1")
    count = len(series) - lookback - horizon + 1
    if count < 1:
        raise ValueError("Series is too short for this lookback and horizon")
    starts = np.arange(count)
    inputs = np.stack([series[s : s + lookback] for s in starts])
    target_index = starts + lookback + horizon - 1
    return inputs, series[target_index], target_index


def split_windows(series: np.ndarray, lookback: int, horizon: int, test_fraction: float) -> dict:
    """Window a series and split it in time order.

    A window belongs to the test set when its target is in the last
    `test_fraction` of the series. Test inputs may reach back into the training
    period, which is allowed: only past values are used to predict.
    """
    inputs, targets, target_index = make_windows(series, lookback, horizon)
    cut = int(len(series) * (1 - test_fraction))
    is_test = target_index >= cut
    if not is_test.any() or is_test.all():
        raise ValueError("test_fraction leaves the training or the test set empty")
    return {
        "x_train": inputs[~is_test],
        "y_train": targets[~is_test],
        "x_test": inputs[is_test],
        "y_test": targets[is_test],
        "train_values": series[:cut],
    }


class MinMaxScaler:
    """Scale values to 0..1 using the minimum and maximum of one client's training data.

    Each client fits its own scaler, so no client has to share its range.
    """

    def __init__(self, train_values: np.ndarray):
        self.low = float(np.min(train_values))
        span = float(np.max(train_values)) - self.low
        self.span = span if span > 0 else 1.0

    def scale(self, values: np.ndarray) -> np.ndarray:
        return (values - self.low) / self.span

    def unscale(self, values: np.ndarray) -> np.ndarray:
        return values * self.span + self.low
