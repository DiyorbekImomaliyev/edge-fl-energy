from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data.loaders import synthetic_clients
from src.data.windows import MinMaxScaler, make_windows, split_windows
from src.evaluation.metrics import mape
from src.federated.fedavg import weighted_average

ROOT = Path(__file__).resolve().parents[1]


def test_weighted_average_uses_sample_counts():
    states = [{"w": np.array([0.0, 2.0])}, {"w": np.array([4.0, 6.0])}]
    averaged = weighted_average(states, sizes=[1, 3])
    assert averaged["w"] == pytest.approx([3.0, 5.0])


def test_weighted_average_rejects_mismatched_input():
    with pytest.raises(ValueError):
        weighted_average([{"w": np.array([1.0])}], sizes=[1, 2])


def test_make_windows_targets_follow_the_horizon():
    series = np.arange(10, dtype=float)
    inputs, targets, index = make_windows(series, lookback=3, horizon=2)
    assert inputs[0].tolist() == [0.0, 1.0, 2.0]
    assert targets[0] == 4.0
    assert index[-1] == 9 and len(inputs) == 6


def test_split_windows_keeps_time_order():
    split = split_windows(np.arange(100, dtype=float), lookback=5, horizon=1, test_fraction=0.2)
    assert split["y_train"].max() < split["y_test"].min()
    assert len(split["y_test"]) == 20
    assert len(split["train_values"]) == 80


def test_scaler_round_trip_and_constant_series():
    values = np.array([2.0, 4.0, 6.0])
    scaler = MinMaxScaler(values)
    assert scaler.scale(values) == pytest.approx([0.0, 0.5, 1.0])
    assert scaler.unscale(scaler.scale(values)) == pytest.approx(values)
    assert MinMaxScaler(np.array([3.0, 3.0])).scale(np.array([3.0])) == pytest.approx([0.0])


def test_mape_skips_zero_targets():
    assert mape(np.array([0.0, 2.0, 4.0]), np.array([1.0, 1.0, 5.0])) == pytest.approx(37.5)


def test_synthetic_clients_are_repeatable_and_different():
    first = synthetic_clients(seed=1, num_clients=3)
    again = synthetic_clients(seed=1, num_clients=3)
    assert list(first) == ["client_00", "client_01", "client_02"]
    assert all(np.array_equal(first[name], again[name]) for name in first)
    assert not np.allclose(first["client_00"], first["client_01"])


def small_federated_config() -> dict:
    config = yaml.safe_load((ROOT / "configs/EXP-000_fedavg_lstm_example.yaml").read_text())
    config["federated"].update(num_clients=3, clients_per_round=2, rounds=2, local_epochs=1, finetune_epochs=1)
    config["baselines"]["local_epochs"] = 2
    config["baselines"]["central_epochs"] = 1
    config["model"]["hidden_size"] = 8
    return config


def test_federated_comparison_reports_every_approach():
    pytest.importorskip("torch")
    from src.federated.comparison import APPROACHES
    from src.pipeline import run

    metrics = run(small_federated_config())
    assert metrics["num_clients"] == 3
    for approach in APPROACHES:
        for metric in ("mae", "rmse", "mape"):
            assert np.isfinite(metrics[f"{approach}_{metric}"])
        assert metrics[f"{approach}_mae"] <= metrics[f"{approach}_rmse"]


def test_federated_comparison_is_repeatable():
    pytest.importorskip("torch")
    from src.pipeline import run

    assert run(small_federated_config()) == run(small_federated_config())


def test_fedavg_changes_the_global_model_and_samples_clients():
    torch = pytest.importorskip("torch")
    from src.federated.fedavg import get_state, run_fedavg
    from src.models.lstm import LSTMForecaster, train_epochs

    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    clients = {}
    for name, series in synthetic_clients(seed=0, num_clients=4, days=5).items():
        split = split_windows(series, lookback=6, horizon=1, test_fraction=0.25)
        clients[name] = {"x_train": split["x_train"], "y_train": split["y_train"]}

    model = LSTMForecaster(hidden_size=4)
    before = get_state(model)
    history = run_fedavg(
        model, clients, rounds=3, clients_per_round=2, local_epochs=1, rng=rng,
        train_fn=lambda m, x, y, epochs: train_epochs(m, x, y, epochs, 0.01, 16, generator),
    )
    after = get_state(model)
    assert len(history) == 3 and all(len(chosen) == 2 for chosen in history)
    assert any(not np.allclose(before[name], after[name]) for name in before)


def test_federated_run_rejects_other_models():
    from src.pipeline import run

    config = small_federated_config()
    config["model"]["type"] = "persistence"
    with pytest.raises(NotImplementedError):
        run(config)


def test_make_windows_with_extra_inputs_puts_load_first():
    series = np.arange(10, dtype=float)
    extra = np.column_stack([series * 10, series * 100])
    inputs, targets, _ = make_windows(series, lookback=3, horizon=1, extra=extra)
    assert inputs.shape == (7, 3, 3)
    assert inputs[0, :, 0].tolist() == [0.0, 1.0, 2.0]
    assert inputs[0, -1].tolist() == [2.0, 20.0, 200.0]
    assert targets[0] == 3.0


def test_calendar_features_are_cyclic():
    import pandas as pd

    from src.data.features import calendar_features

    features = calendar_features(pd.date_range("2013-01-07", periods=24 * 8, freq="h"))  # starts on a Monday
    assert features.shape == (192, 4)
    assert np.abs(features).max() <= 1.0
    assert features[0] == pytest.approx(features[24 * 7])   # same hour, one week later
    assert features[0, :2] == pytest.approx(features[24, :2])  # same hour, next day
    assert not np.allclose(features[0, 2:], features[24, 2:])  # but a different weekday


def test_validation_split_never_uses_the_test_period():
    from src.federated.comparison import evaluation_split

    assert evaluation_split({"test_fraction": 0.25}, 1000) == (1000, 0.25)
    usable, fraction = evaluation_split(
        {"test_fraction": 0.25, "evaluate_on": "validation", "validation_fraction": 0.15}, 1000
    )
    assert usable == 750
    assert fraction == pytest.approx(0.2)  # 150 of the 750 usable hours


def test_federated_comparison_with_calendar_inputs_and_fewer_approaches():
    pytest.importorskip("torch")
    from src.pipeline import run

    config = small_federated_config()
    config["data"]["calendar_features"] = True
    config["approaches"] = ["persistence", "fedavg"]
    metrics = run(config)
    assert np.isfinite(metrics["fedavg_mae"])
    assert "local_only_mae" not in metrics and "centralised_mae" not in metrics


def test_federated_comparison_rejects_unknown_approaches():
    pytest.importorskip("torch")
    from src.pipeline import run

    config = small_federated_config()
    config["approaches"] = ["fedprox"]
    with pytest.raises(ValueError):
        run(config)


def test_ward_clusters_separates_two_obvious_groups():
    from src.federated.clustering import ward_clusters

    rng = np.random.default_rng(0)
    points = np.concatenate([rng.normal(0, 0.1, (5, 3)), rng.normal(5, 0.1, (4, 3))])
    labels = ward_clusters(points, num_clusters=2)
    assert labels.tolist() == [0] * 5 + [1] * 4
    assert ward_clusters(points, num_clusters=9).tolist() == list(range(9))
    with pytest.raises(ValueError):
        ward_clusters(points, num_clusters=10)


def test_clustered_fedavg_reports_scores_and_keeps_other_approaches_unchanged():
    pytest.importorskip("torch")
    from src.pipeline import run

    plain = small_federated_config()
    plain["federated"]["num_clients"] = 6
    plain["approaches"] = ["persistence", "fedavg"]

    clustered = small_federated_config()
    clustered["federated"]["num_clients"] = 6
    clustered["federated"]["clustering"] = {"after_rounds": 1, "num_clusters": 2}
    clustered["approaches"] = ["persistence", "fedavg", "fedavg_clustered", "fedavg_clustered_finetuned"]

    with_clusters = run(clustered)
    assert with_clusters["fedavg_mae"] == run(plain)["fedavg_mae"]
    assert np.isfinite(with_clusters["fedavg_clustered_mae"])
    assert np.isfinite(with_clusters["fedavg_clustered_finetuned_rmse"])
    assert sum(int(size) for size in with_clusters["cluster_sizes"].split("/")) == 6
