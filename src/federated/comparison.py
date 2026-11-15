"""Compare FedAvg with the baselines it has to beat, on the same clients and test data.

Approaches, each scored on every client's own test set:
- persistence:       repeat the last known value (no training)
- local_only:        each client trains its own model on its own data
- fedavg:            one shared model trained with federated averaging
- fedavg_finetuned:  the shared model, then a few local epochs on each client
"""
import copy

import numpy as np
import torch

from src.data.loaders import load_clients
from src.data.windows import MinMaxScaler, split_windows
from src.evaluation.metrics import mae, mape, rmse
from src.federated.fedavg import run_fedavg
from src.models.lstm import LSTMForecaster, predict, train_epochs

APPROACHES = ("persistence", "local_only", "fedavg", "fedavg_finetuned")


def prepare_clients(config: dict) -> dict[str, dict]:
    """Load every client's series, window it, split it in time order and scale it per client."""
    data_cfg = config["data"]
    clients = {}
    for name, series in load_clients(data_cfg, config["seed"], config["federated"]["num_clients"]).items():
        split = split_windows(series, data_cfg["lookback"], data_cfg.get("horizon", 1), data_cfg["test_fraction"])
        scaler = MinMaxScaler(split["train_values"])
        clients[name] = {
            "x_train": scaler.scale(split["x_train"]),
            "y_train": scaler.scale(split["y_train"]),
            "x_test": scaler.scale(split["x_test"]),
            "y_test_raw": split["y_test"],
            "last_value_raw": split["x_test"][:, -1],
            "scaler": scaler,
        }
    return clients


def run_comparison(config: dict) -> dict:
    """Run all approaches and return the mean of each metric over clients, in the data's own units."""
    model_cfg, fed_cfg = config["model"], config["federated"]
    if fed_cfg.get("strategy", "fedavg") != "fedavg":
        raise NotImplementedError(f"Strategy '{fed_cfg['strategy']}' is not available. Only fedavg is implemented.")

    seed = config["seed"]
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    clients = prepare_clients(config)

    def train_fn(model, inputs, targets, epochs):
        train_epochs(model, inputs, targets, epochs, model_cfg["learning_rate"], model_cfg["batch_size"], generator)

    # Every approach starts from the same initial weights.
    initial_model = LSTMForecaster(model_cfg["hidden_size"], model_cfg["num_layers"])

    global_model = copy.deepcopy(initial_model)
    run_fedavg(
        global_model, clients, fed_cfg["rounds"], fed_cfg["clients_per_round"], fed_cfg["local_epochs"], train_fn, rng
    )

    scores = {approach: {"mae": [], "rmse": [], "mape": []} for approach in APPROACHES}
    for name in sorted(clients):
        data = clients[name]
        truth = data["y_test_raw"]

        local_model = copy.deepcopy(initial_model)
        train_fn(local_model, data["x_train"], data["y_train"], config["baselines"]["local_epochs"])

        tuned_model = copy.deepcopy(global_model)
        train_fn(tuned_model, data["x_train"], data["y_train"], fed_cfg["finetune_epochs"])

        predictions = {
            "persistence": data["last_value_raw"],
            "local_only": data["scaler"].unscale(predict(local_model, data["x_test"])),
            "fedavg": data["scaler"].unscale(predict(global_model, data["x_test"])),
            "fedavg_finetuned": data["scaler"].unscale(predict(tuned_model, data["x_test"])),
        }
        for approach, predicted in predictions.items():
            scores[approach]["mae"].append(mae(truth, predicted))
            scores[approach]["rmse"].append(rmse(truth, predicted))
            scores[approach]["mape"].append(mape(truth, predicted))

    metrics = {"num_clients": len(clients), "test_size_per_client": int(len(next(iter(clients.values()))["y_test_raw"]))}
    for approach in APPROACHES:
        for metric, values in scores[approach].items():
            metrics[f"{approach}_{metric}"] = float(np.mean(values))
    return metrics
