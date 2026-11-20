import numpy as np
import pandas as pd
import pytest

from src.data.pecan import hourly_homes, read_pecan_csv, read_raw, week_fill


def raw_rows(home: int, hours: int, grid: float, solar: str = "", skip: tuple[int, ...] = ()) -> str:
    """Quarter-hour rows from 2018-07-01 in the raw Dataport layout (extra circuit column included)."""
    times = pd.date_range("2018-07-01", periods=hours * 4, freq="15min")
    lines = []
    for index, time in enumerate(times):
        if index not in skip:
            lines.append(f"{home},{time:%Y-%m-%d %H:%M:%S}-05,0.3,{grid},{solar},\n")
    return "".join(lines)


@pytest.fixture
def raw_file(tmp_path):
    text = "dataid,local_15min,air1,grid,solar,solar2\n"
    text += raw_rows(11, hours=48, grid=1.0)                       # no panels
    text += raw_rows(22, hours=48, grid=-0.5, solar="2.0")          # exporting: use is 1.5 kW
    text += raw_rows(33, hours=48, grid=1.0, skip=(40,))            # one missing hour: filled
    text += raw_rows(44, hours=48, grid=1.0, skip=tuple(range(40, 80)))  # ten missing hours: dropped
    text += raw_rows(55, hours=48, grid=-1.0)                       # negative use: dropped
    path = tmp_path / "15minute_data_test.csv"
    path.write_text(text)
    return path


def test_read_pecan_csv_adds_solar_to_grid_and_drops_the_utc_offset(raw_file):
    frame = read_pecan_csv(raw_file)
    assert list(frame.columns) == ["home", "time", "use_kw"]
    assert frame["time"].iloc[1] == pd.Timestamp("2018-07-01 00:15:00")
    assert frame.loc[frame["home"] == "11", "use_kw"].iloc[0] == pytest.approx(1.0)
    assert frame.loc[frame["home"] == "22", "use_kw"].iloc[0] == pytest.approx(1.5)


def test_hourly_homes_gives_kwh_per_hour_and_keeps_only_complete_sane_homes(raw_file):
    homes, dropped = hourly_homes(read_pecan_csv(raw_file), "2018-07-01", "2018-07-03")
    assert sorted(homes) == ["11", "22", "33"]
    assert len(homes["11"]) == 48
    assert homes["11"].to_numpy() == pytest.approx(np.full(48, 1.0))
    assert homes["22"].to_numpy() == pytest.approx(np.full(48, 1.5))
    assert homes["33"].iloc[10] == pytest.approx(1.0)  # the hour with a missing reading was interpolated
    assert "missing" in dropped["44"]
    assert "negative" in dropped["55"]


def test_hourly_homes_without_gap_filling_drops_the_home_with_a_missing_reading(raw_file):
    homes, _ = hourly_homes(read_pecan_csv(raw_file), "2018-07-01", "2018-07-03", max_gap_hours=0)
    assert sorted(homes) == ["11", "22"]


def test_read_raw_reads_a_folder_and_skips_metadata(raw_file):
    (raw_file.parent / "metadata.csv").write_text("dataid,city\n11,Austin\n")
    frame = read_raw(raw_file.parent)
    assert frame["home"].nunique() == 5


def test_week_fill_copies_the_week_before_and_leaves_long_gaps():
    hours = pd.date_range("2018-07-01", periods=24 * 21, freq="h")
    series = pd.Series(np.arange(len(hours), dtype=float), index=hours)
    series.iloc[200:210] = np.nan   # 10 hours: filled from 168 hours earlier
    series.iloc[400:440] = np.nan   # 40 hours: too long, stays missing
    series.iloc[5:8] = np.nan       # first week: filled from the week after
    filled = week_fill(series, max_hours=24)
    assert filled.iloc[200:210].to_numpy() == pytest.approx(np.arange(32, 42))
    assert filled.iloc[400:440].isna().all()
    assert filled.iloc[5:8].to_numpy() == pytest.approx(np.arange(173, 176))
