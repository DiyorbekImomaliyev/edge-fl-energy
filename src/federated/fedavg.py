"""Federated averaging (FedAvg, McMahan et al. 2017), run in one process.

Each client trains a copy of the global model on its own data. Only model
weights are passed to the server, never the data. This file simulates the
clients one after another; it does not measure real communication.
"""
import copy

import numpy as np


def weighted_average(states: list[dict[str, np.ndarray]], sizes: list[int]) -> dict[str, np.ndarray]:
    """Average client weights, giving each client a share equal to its number of samples."""
    if not states or len(states) != len(sizes):
        raise ValueError("Need one sample count per client state")
    total = float(sum(sizes))
    if total <= 0:
        raise ValueError("Clients have no samples")
    return {name: sum(state[name] * (size / total) for state, size in zip(states, sizes)) for name in states[0]}


def get_state(model) -> dict[str, np.ndarray]:
    """Copy a torch model's weights into numpy arrays."""
    return {name: tensor.detach().cpu().numpy().copy() for name, tensor in model.state_dict().items()}


def set_state(model, state: dict[str, np.ndarray]) -> None:
    """Load numpy weights into a torch model."""
    import torch

    model.load_state_dict({name: torch.as_tensor(value) for name, value in state.items()})


def run_fedavg(
    global_model,
    clients: dict[str, dict],
    rounds: int,
    clients_per_round: int,
    local_epochs: int,
    train_fn,
    rng: np.random.Generator,
) -> list[list[str]]:
    """Train `global_model` in place with FedAvg and return the clients chosen in each round.

    `clients` maps a client name to its data dict (`x_train`, `y_train`).
    `train_fn(model, inputs, targets, epochs)` trains a model on one client's data.
    """
    names = sorted(clients)
    if not 1 <= clients_per_round <= len(names):
        raise ValueError("clients_per_round must be between 1 and the number of clients")
    history = []
    for _ in range(rounds):
        chosen = sorted(rng.choice(names, size=clients_per_round, replace=False).tolist())
        states, sizes = [], []
        for name in chosen:
            local_model = copy.deepcopy(global_model)
            data = clients[name]
            train_fn(local_model, data["x_train"], data["y_train"], local_epochs)
            states.append(get_state(local_model))
            sizes.append(len(data["y_train"]))
        set_state(global_model, weighted_average(states, sizes))
        history.append(chosen)
    return history
