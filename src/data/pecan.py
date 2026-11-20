"""Prepare the Pecan Street Dataport data: one hourly CSV file per household.

Source: Pecan Street Dataport, free university access (https://dataport.pecanstreet.org,
"Residential Data" page). The 15-minute files hold one row per home and quarter hour,
with the average power in kW on each circuit. The data may not be shared outside the
licence, so it must stay in data/, which Git ignores.

How the whole-home load is built:
- `grid` is the power drawn from the grid. It is negative when a home with solar panels
  sends power out. `solar` and `solar2` are the power the panels produce.
- Household use = grid + solar + solar2. For homes without panels this is just `grid`.
- An hour's energy in kWh is the mean of its four quarter-hour kW readings.
"""
from pathlib import Path

import pandas as pd

POWER_COLUMNS = ["grid", "solar", "solar2"]


def read_pecan_csv(source) -> pd.DataFrame:
    """Read one raw 15-minute file (plain or .gz) into the columns home, time, use_kw.

    `local_15min` looks like `2018-11-21 15:15:00-06`. The UTC offset at the end is cut
    off, so times are local clock times. A missing `grid` reading gives a missing use;
    a missing `solar` reading counts as zero (most homes have no panels).
    """
    frame = pd.read_csv(source, usecols=lambda name: name in ["dataid", "local_15min", *POWER_COLUMNS])
    for column in POWER_COLUMNS:
        if column not in frame.columns:
            frame[column] = float("nan")
    use = frame["grid"] + frame["solar"].fillna(0) + frame["solar2"].fillna(0)
    out = pd.DataFrame(
        {
            "home": frame["dataid"].astype(str),
            "time": pd.to_datetime(frame["local_15min"].astype(str).str.slice(0, 19), errors="coerce"),
            "use_kw": use,
        }
    )
    return out.dropna(subset=["time"])


def hourly_homes(
    frame: pd.DataFrame,
    start: str,
    end: str,
    max_gap_hours: int = 2,
    max_negative_share: float = 0.01,
    max_week_fill_hours: int = 0,
) -> tuple[dict[str, pd.Series], dict[str, str]]:
    """Return one complete hourly series (kWh) per home for [start, end), and why homes were left out.

    - An hour is the mean of its four quarter-hour readings. If any is missing, the hour is missing.
    - Use cannot be negative. A home where more than `max_negative_share` of the readings are
      below -0.01 kW is left out (its solar or grid channel is probably wired or signed wrongly).
      Other negative readings are set to zero.
    - Gaps of up to `max_gap_hours` hours are filled by straight-line interpolation.
    - Longer gaps of up to `max_week_fill_hours` hours are filled with the same hours one week
      earlier (one week later if that is missing too). 0 turns this off.
    - A home that still has a missing hour is left out, so every home has the same hours.
    """
    start_time, end_time = pd.Timestamp(start), pd.Timestamp(end)
    hours = pd.date_range(start_time, end_time, freq="h", inclusive="left")
    rows = frame[(frame["time"] >= start_time) & (frame["time"] < end_time)]

    homes, dropped = {}, {}
    for home, group in rows.groupby("home", sort=True):
        readings = group.groupby("time")["use_kw"].mean()  # a repeated clock time (clock change) is averaged
        negative_share = float((readings < -0.01).mean())
        if negative_share > max_negative_share:
            dropped[home] = f"{negative_share:.1%} of readings are negative"
            continue
        readings = readings.clip(lower=0)
        counts = readings.resample("h").count()
        hourly = readings.resample("h").mean().where(counts == 4).reindex(hours)
        if max_gap_hours > 0:
            hourly = hourly.interpolate(limit=max_gap_hours, limit_area="inside")
        if max_week_fill_hours > 0:
            hourly = week_fill(hourly, max_week_fill_hours)
        if hourly.notna().all():
            homes[home] = hourly
        else:
            dropped[home] = f"{int(hourly.isna().sum())} hours still missing after gap filling"
    return homes, dropped


def week_fill(hourly: pd.Series, max_hours: int) -> pd.Series:
    """Fill each run of missing hours no longer than `max_hours` with the values 168 hours earlier.

    Hours one week later are used where the earlier week is missing as well. Longer runs stay missing.
    """
    missing = hourly.isna()
    run_id = (missing != missing.shift()).cumsum()
    run_length = missing.groupby(run_id).transform("sum")
    fillable = missing & (run_length <= max_hours)
    donor = hourly.shift(168).fillna(hourly.shift(-168))
    return hourly.where(~fillable, donor)


def read_raw(source: Path, pattern: str = "*.csv*") -> pd.DataFrame:
    """Read one raw file, or every file in a folder whose name matches `pattern`.

    `metadata.csv` is skipped. Puerto Rico comes as one file per month and measurement:
    use the pattern `pr_realpower_*` for it.
    """
    if source.is_dir():
        paths = sorted(p for p in source.glob(pattern) if p.name != "metadata.csv")
    else:
        paths = [source]
    if not paths:
        raise ValueError(f"No files matching {pattern} found in {source}")
    return pd.concat([read_pecan_csv(path) for path in paths], ignore_index=True)
