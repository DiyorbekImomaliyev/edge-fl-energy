"""Extra model inputs that are known without measuring anything, such as the time of day."""
import numpy as np
import pandas as pd

CALENDAR_FEATURES = 4


def calendar_features(times: pd.DatetimeIndex) -> np.ndarray:
    """Hour of day and day of week for each timestamp, each as a sine and cosine pair.

    Sine and cosine keep the cycle joined up: 23:00 is next to 00:00, and Sunday is next to Monday.
    Returns shape (len(times), 4) with values between -1 and 1.
    """
    hour = 2 * np.pi * (times.hour + times.minute / 60) / 24
    day = 2 * np.pi * times.day_of_week / 7
    return np.column_stack([np.sin(hour), np.cos(hour), np.sin(day), np.cos(day)])
