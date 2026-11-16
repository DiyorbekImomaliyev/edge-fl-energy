import numpy as np
import pandas as pd
import pytest

from src.data.lcl import choose_homes, hourly_homes, read_lcl_csv, read_raw, write_client_files
from src.data.loaders import load_clients

HEADER = "LCLid,stdorToU,DateTime,KWH/hh (per half hour) \n"


def raw_rows(home: str, tariff: str, hours: int, skip: tuple[int, ...] = (), null: tuple[int, ...] = ()) -> str:
    """Half-hourly rows from 2013-01-01 in the raw LCL layout. Each half hour uses 0.25 kWh."""
    times = pd.date_range("2013-01-01", periods=hours * 2, freq="30min")
    lines = []
    for index, time in enumerate(times):
        if index in skip:
            continue
        value = "Null" if index in null else " 0.25 "
        lines.append(f"{home},{tariff},{time:%Y-%m-%d %H:%M:%S}.0000000,{value}\n")
    return "".join(lines)


@pytest.fixture
def raw_file(tmp_path):
    text = HEADER
    text += raw_rows("MAC000001", "Std", hours=48)
    text += raw_rows("MAC000002", "Std", hours=48, null=(10,))            # one missing hour: filled
    text += raw_rows("MAC000003", "Std", hours=48, skip=tuple(range(20, 40)))  # ten missing hours: dropped
    text += raw_rows("MAC000004", "ToU", hours=48)                        # wrong tariff: dropped
    path = tmp_path / "LCL-test.csv"
    path.write_text(text)
    return path


def test_read_lcl_csv_parses_times_and_null_readings(raw_file):
    frame = read_lcl_csv(raw_file)
    assert list(frame.columns) == ["home", "tariff", "time", "kwh"]
    assert frame["time"].iloc[1] == pd.Timestamp("2013-01-01 00:30:00")
    assert frame["kwh"].isna().sum() == 1
    assert frame["kwh"].max() == pytest.approx(0.25)


def test_hourly_homes_sums_half_hours_and_keeps_only_complete_std_homes(raw_file):
    homes = hourly_homes(read_lcl_csv(raw_file), "2013-01-01", "2013-01-03")
    assert sorted(homes) == ["MAC000001", "MAC000002"]
    assert len(homes["MAC000001"]) == 48
    assert homes["MAC000001"].to_numpy() == pytest.approx(np.full(48, 0.5))
    assert homes["MAC000002"].iloc[5] == pytest.approx(0.5)  # the hour with a Null half was interpolated


def test_hourly_homes_without_gap_filling_drops_the_home_with_a_missing_hour(raw_file):
    homes = hourly_homes(read_lcl_csv(raw_file), "2013-01-01", "2013-01-03", max_gap_hours=0)
    assert sorted(homes) == ["MAC000001"]


def test_hourly_homes_can_keep_every_tariff(raw_file):
    homes = hourly_homes(read_lcl_csv(raw_file), "2013-01-01", "2013-01-03", tariff=None)
    assert "MAC000004" in homes


def test_choose_homes_is_repeatable_and_checks_the_count():
    homes = {f"MAC{i:06d}": None for i in range(10)}
    assert choose_homes(homes, 4, seed=1) == choose_homes(homes, 4, seed=1)
    assert len(set(choose_homes(homes, 4, seed=1))) == 4
    with pytest.raises(ValueError):
        choose_homes(homes, 11, seed=1)


def test_prepared_files_load_as_federated_clients(raw_file, tmp_path, monkeypatch):
    frame = read_raw(raw_file, "2013-01-01", "2013-01-03")
    homes = hourly_homes(frame, "2013-01-01", "2013-01-03")
    paths = write_client_files(homes, sorted(homes), tmp_path / "out", "D001_lcl")
    assert [p.name for p in paths] == ["D001_lcl_MAC000001.csv", "D001_lcl_MAC000002.csv"]

    monkeypatch.chdir(tmp_path)
    clients = load_clients({"source": "csv", "path_glob": "out/D001_lcl_*.csv", "target_column": "load_kwh"}, 0, 0)
    assert sorted(clients) == ["D001_lcl_MAC000001", "D001_lcl_MAC000002"]
    assert clients["D001_lcl_MAC000001"].shape == (48,)


def test_read_raw_reads_a_zip_and_filters_the_period(raw_file, tmp_path):
    import zipfile

    archive = tmp_path / "raw.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(raw_file, "Partitioned/LCL-test.csv")
    frame = read_raw(archive, "2013-01-01", "2013-01-02")
    assert frame["time"].max() < pd.Timestamp("2013-01-02")
    assert frame["home"].nunique() == 4


def test_csv_clients_need_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        load_clients({"source": "csv", "path_glob": "nothing/*.csv", "target_column": "load_kwh"}, 0, 0)
