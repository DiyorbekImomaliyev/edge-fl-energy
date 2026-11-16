"""Prepare the Low Carbon London smart meter data: one hourly CSV file per household.

Source: "SmartMeter Energy Consumption Data in London Households", UK Power Networks,
published on the London Datastore under CC BY 4.0. The raw files hold one row per
household and half hour, with the energy used in that half hour in kWh.

The preparation follows Briggs, Fan and Andras (2022): flat-rate (Std) tariff homes
only, one fixed period, half-hourly readings summed to hourly values.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["home", "tariff", "time", "kwh"]


def read_lcl_csv(source) -> pd.DataFrame:
    """Read one raw LCL file (a path or an open file) into the columns home, tariff, time, kwh.

    The raw header is `LCLid, stdorToU, DateTime, KWH/hh (per half hour)`. Columns are
    taken by position because the last header has stray spaces. Missing readings are
    written as the text "Null" and become NaN.
    """
    frame = pd.read_csv(source, header=0, names=COLUMNS, usecols=[0, 1, 2, 3], dtype=str)
    frame["time"] = pd.to_datetime(frame["time"].str.slice(0, 19), errors="coerce")
    frame["kwh"] = pd.to_numeric(frame["kwh"], errors="coerce")
    return frame.dropna(subset=["time"])


def hourly_homes(
    frame: pd.DataFrame,
    start: str,
    end: str,
    tariff: str | None = "Std",
    max_gap_hours: int = 2,
) -> dict[str, pd.Series]:
    """Return one complete hourly series per household for the period [start, end).

    - Only homes on `tariff` are kept (None keeps all).
    - An hour is the sum of its two half-hour readings. If either is missing, the hour is missing.
    - Gaps of up to `max_gap_hours` hours are filled by straight-line interpolation.
    - A home that still has a missing hour is left out, so every home has the same hours.
    """
    start_time, end_time = pd.Timestamp(start), pd.Timestamp(end)
    hours = pd.date_range(start_time, end_time, freq="h", inclusive="left")
    rows = frame[(frame["time"] >= start_time) & (frame["time"] < end_time)]
    if tariff is not None:
        rows = rows[rows["tariff"] == tariff]

    homes = {}
    for home, group in rows.groupby("home", sort=True):
        readings = group.drop_duplicates("time").set_index("time")["kwh"]
        hourly = readings.resample("h").sum(min_count=2).reindex(hours)
        if max_gap_hours > 0:
            hourly = hourly.interpolate(limit=max_gap_hours, limit_area="inside")
        if hourly.notna().all():
            homes[home] = hourly
    return homes


def choose_homes(homes: dict[str, pd.Series], count: int, seed: int) -> list[str]:
    """Pick `count` homes at random, repeatably for a given seed."""
    names = sorted(homes)
    if count > len(names):
        raise ValueError(f"Asked for {count} homes but only {len(names)} are complete for this period")
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(names, size=count, replace=False).tolist())


def write_client_files(homes: dict[str, pd.Series], names: list[str], out_dir: Path, prefix: str) -> list[Path]:
    """Write one CSV per home with the columns time and load_kwh. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = out_dir / f"{prefix}_{name}.csv"
        homes[name].rename("load_kwh").rename_axis("time").to_csv(path)
        paths.append(path)
    return paths


def read_raw(source: Path, start: str, end: str) -> pd.DataFrame:
    """Read every raw CSV in a folder or a zip file, keeping only rows inside [start, end).

    Filtering file by file keeps memory use low: the full dataset has about 167 million rows.
    """
    start_time, end_time = pd.Timestamp(start), pd.Timestamp(end)

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[(frame["time"] >= start_time) & (frame["time"] < end_time)]

    parts = []
    if source.is_dir():
        for path in sorted(source.rglob("*.csv")):
            parts.append(keep(read_lcl_csv(path)))
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for member in sorted(n for n in archive.namelist() if n.lower().endswith(".csv")):
                with archive.open(member) as handle:
                    parts.append(keep(read_lcl_csv(handle)))
    else:
        parts.append(keep(read_lcl_csv(source)))
    if not parts:
        raise ValueError(f"No CSV files found in {source}")
    return pd.concat(parts, ignore_index=True)
