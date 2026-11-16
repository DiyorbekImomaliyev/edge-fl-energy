"""Compare FedAvg with the baselines it has to beat, on the same clients and test data.

Approaches, each scored on every client's own test set:
- persistence:       repeat the last known value (no training)
- local_only:        each client trains its own model on its own data
- centralised:       one model trained on all clients' data in one place (not private)
- fedavg:            one shared model trained with federated averaging
- fedavg_finetuned:  the shared model, then a few local epochs on each client

A config can list fewer approaches under `approaches` to save time, for example while tuning.
"""
import copy

import numpy as np
import torch

from src.data.features import CALENDAR_FEATURES, calendar_features
from src.data.loaders import load_clients_with_time
from src.data.windows import MinMaxScaler, split_windows
from src.evaluation.metrics import mae, mape, rmse
from src.federated.fedavg import run_fedavg
from src.models.lstm import LSTMForecaster, predict, train_epochs

APPROACHES = ("persistence", "local_only", "centralised", "fedavg", "fedavg_finetuned")


def evaluation_split(data_cfg: dict, length: int) -> tuple[int, float]:
    """Return how much of a series to use and which fraction of that part is scored.

    - `evaluate_on: test` (default): the whole series; the last `test_fraction` is scored.
    - `evaluate_on: validation`: the test period is cut off and never seen. The last
      `validation_fraction` of the whole series before the test period is scored instead.
      Use this to choose settings, so the test set stays untouched.
    """
    test_fraction = data_cfg["test_fraction"]
    mode = data_cfg.get("evaluate_on", "test")
    if mode == "test":
        return length, test_fraction
    if mode == "validation":
        validation_fraction = data_cfg["validation_fraction"]
        usable = int(length * (1 - test_fraction))
        return usable, validation_fraction / (1 - test_fraction)
    raise ValueError(f"evaluate_on must be 'test' or 'validation', not '{mode}'")


def prepare_clients(config: dict) -> dict[str, dict]:
    """Load every client's series, split it in time order, scale it per client and window it."""
    data_cfg = config["data"]
    lookback, horizon = data_cfg["lookback"], data_cfg.get("horizon", 1)
    use_calendar = bool(data_cfg.get("calendar_features", False))
    clients = {}
    loaded = load_clients_with_time(data_cfg, config["seed"], config["federated"]["num_clients"])
    for name, (series, times) in loaded.items():
        usable, fraction = evaluation_split(data_cfg, len(series))
        series, times = series[:usable], times[:usable]
        extra = calendar_features(times) if use_calendar else None

        raw = split_windows(series, lookback, horizon, fraction)
        scaler = MinMaxScaler(raw["train_values"])
        scaled = split_windows(scaler.scale(series), lookback, horizon, fraction, extra)
        clients[name] = {
            "x_train": scaled["x_train"],
            "y_train": scaled["y_train"],
            "x_test": scaled["x_test"],
            "y_test_raw": raw["y_test"],
            "last_value_raw": raw["x_test"][:, -1],  # the newest value the model is allowed to see
            "scaler": scaler,
        }
    return clients


def run_comparison(config: dict) -> dict:
    """Run the chosen approaches and return the mean of each metric over clients, in the data's own units."""
    model_cfg, fed_cfg = config["model"], config["federated"]
    if fed_cfg.get("strategy", "fedavg") != "fedavg":
        raise NotImplementedError(f"Strategy '{fed_cfg['strategy']}' is not available. Only fedavg is implemented.")
    approaches = tuple(config.get("approaches") or APPROACHES)
    unknown = set(approaches) - set(APPROACHES)
    if unknown:
        raise ValueError(f"Unknown approaches: {sorted(unknown)}")

    seed = config["seed"]
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    clients = prepare_clients(config)
    names = sorted(clients)

    def train_fn(model, inputs, targets, epochs):
        train_epochs(model, inputs, targets, epochs, model_cfg["learning_rate"], model_cfg["batch_size"], generator)

    # Every approach starts from the same initial weights.
    input_size = 1 + (CALENDAR_FEATURES if config["data"].get("calendar_features") else 0)
    initial_model = LSTMForecaster(model_cfg["hidden_size"], model_cfg["num_layers"], input_size)

    global_model = None
    if "fedavg" in approaches or "fedavg_finetuned" in approaches:
        global_model = copy.deepcopy(initial_model)
        run_fedavg(
            global_model, clients, fed_cfg["rounds"], fed_cfg["clients_per_round"], fed_cfg["local_epochs"], train_fn, rng
        )

    central_model = None
    if "centralised" in approaches:
        # Every client's (already scaled) training windows pooled in one place.
        central_model = copy.deepcopy(initial_model)
        pooled_x = np.concatenate([clients[name]["x_train"] for name in names])
        pooled_y = np.concatenate([clients[name]["y_train"] for name in names])
        train_fn(central_model, pooled_x, pooled_y, config["baselines"]["central_epochs"])

    scores = {approach: {"mae": [], "rmse": [], "mape": []} for approach in approaches}
    for name in names:
        data = clients[name]
        truth = data["y_test_raw"]

        def forecast(model):
            return data["scaler"].unscale(predict(model, data["x_test"]))

        predictions = {}
        if "persistence" in approaches:
            predictions["persistence"] = data["last_value_raw"]
        if "local_only" in approaches:
            local_model = copy.deepcopy(initial_model)
            train_fn(local_model, data["x_train"], data["y_train"], config["baselines"]["local_epochs"])
            predictions["local_only"] = forecast(local_model)
        if "centralised" in approaches:
            predictions["centralised"] = forecast(central_model)
        if "fedavg" in approaches:
            predictions["fedavg"] = forecast(global_model)
        if "fedavg_finetuned" in approaches:
            tuned_model = copy.deepcopy(global_model)
            train_fn(tuned_model, data["x_train"], data["y_train"], fed_cfg["finetune_epochs"])
            predictions["fedavg_finetuned"] = forecast(tuned_model)

        for approach, predicted in predictions.items():
            scores[approach]["mae"].append(mae(truth, predicted))
            scores[approach]["rmse"].append(rmse(truth, predicted))
            scores[approach]["mape"].append(mape(truth, predicted))

    metrics = {"num_clients": len(clients), "test_size_per_client": int(len(clients[names[0]]["y_test_raw"]))}
    for approach in approaches:
        for metric, values in scores[approach].items():
            metrics[f"{approach}_{metric}"] = float(np.mean(values))
    return metrics
