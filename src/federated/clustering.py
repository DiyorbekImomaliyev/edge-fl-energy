"""Group clients by how they change the shared model, without looking at their data.

Idea from Briggs, Fan and Andras (2022), "FL+HC": if two homes pull the model in a similar
direction, they probably use energy in a similar way. The server only sees model updates,
so no home shares readings or summary statistics.
"""
import copy

import numpy as np

from src.federated.fedavg import get_state, run_fedavg


def ward_clusters(points: np.ndarray, num_clusters: int) -> np.ndarray:
    """Hierarchical clustering with Ward linkage. Returns a cluster number (0, 1, ...) per row.

    Starts with every row as its own cluster and keeps joining the two clusters whose
    merge adds the least spread, until `num_clusters` are left. Clusters are numbered in
    the order of their first member, so the result is repeatable.
    """
    count = len(points)
    if not 1 <= num_clusters <= count:
        raise ValueError("num_clusters must be between 1 and the number of points")
    members = {index: [index] for index in range(count)}
    centres = {index: points[index].astype(float) for index in range(count)}
    while len(members) > num_clusters:
        best = None
        keys = sorted(members)
        for position, first in enumerate(keys):
            for second in keys[position + 1 :]:
                size_a, size_b = len(members[first]), len(members[second])
                gap = centres[first] - centres[second]
                cost = size_a * size_b / (size_a + size_b) * float(gap @ gap)
                if best is None or cost < best[0]:
                    best = (cost, first, second)
        _, first, second = best
        size_a, size_b = len(members[first]), len(members[second])
        centres[first] = (centres[first] * size_a + centres[second] * size_b) / (size_a + size_b)
        members[first] += members.pop(second)
        del centres[second]
    labels = np.empty(count, dtype=int)
    for number, key in enumerate(sorted(members, key=lambda k: min(members[k]))):
        labels[members[key]] = number
    return labels


def client_updates(global_model, clients: dict[str, dict], names: list[str], local_epochs: int, train_fn) -> np.ndarray:
    """Train a copy of the global model on each client and return each change as one flat row."""
    start = get_state(global_model)
    rows = []
    for name in names:
        local_model = copy.deepcopy(global_model)
        train_fn(local_model, clients[name]["x_train"], clients[name]["y_train"], local_epochs)
        after = get_state(local_model)
        rows.append(np.concatenate([(after[key] - start[key]).ravel() for key in sorted(start)]))
    return np.stack(rows)


def run_clustered_fedavg(
    global_model,
    clients: dict[str, dict],
    rounds: int,
    clients_per_round: int,
    local_epochs: int,
    train_fn,
    rng: np.random.Generator,
    after_rounds: int,
    num_clusters: int,
) -> tuple[dict[str, object], dict[str, int]]:
    """FedAvg for `after_rounds` rounds, then cluster the clients and continue FedAvg inside each cluster.

    Returns (model for each client, cluster number for each client). `global_model` is trained
    in place for the first stage only.

    After clustering, each cluster picks a share of `clients_per_round` equal to its share of
    the clients (at least one), so a round costs about the same as a plain FedAvg round.
    Working out the updates needs one extra local training on every client.
    """
    if not 0 <= after_rounds <= rounds:
        raise ValueError("after_rounds must be between 0 and rounds")
    names = sorted(clients)
    run_fedavg(global_model, clients, after_rounds, clients_per_round, local_epochs, train_fn, rng)
    updates = client_updates(global_model, clients, names, local_epochs, train_fn)
    labels = ward_clusters(updates, num_clusters)

    models, cluster_of = {}, {}
    for cluster in range(num_clusters):
        group = [name for name, label in zip(names, labels) if label == cluster]
        share = max(1, round(clients_per_round * len(group) / len(names)))
        cluster_model = copy.deepcopy(global_model)
        run_fedavg(
            cluster_model,
            {name: clients[name] for name in group},
            rounds - after_rounds,
            min(share, len(group)),
            local_epochs,
            train_fn,
            rng,
        )
        for name in group:
            models[name] = cluster_model
            cluster_of[name] = cluster
    return models, cluster_of
