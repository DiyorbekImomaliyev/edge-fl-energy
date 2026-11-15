"""Run one experiment: load data, forecast, evaluate.

With `federated.enabled: true` the run compares FedAvg with its baselines; see src/federated/comparison.py.
"""
from src.data.loaders import load_series, train_test_split
from src.evaluation.metrics import mae, rmse
from src.models.baselines import persistence_forecast

# Add new models here: "name": function(train, test) -> predictions
MODELS = {"persistence": persistence_forecast}


def run(config: dict) -> dict:
    """Run the experiment described by `config` and return its metrics."""
    model_type = config["model"]["type"]
    if config.get("federated", {}).get("enabled"):
        if model_type != "lstm":
            raise NotImplementedError(f"Federated training supports model 'lstm' only, not '{model_type}'.")
        # Imported here so that non-federated runs do not need PyTorch.
        from src.federated.comparison import run_comparison

        return run_comparison(config)
    if model_type not in MODELS:
        raise NotImplementedError(
            f"Model '{model_type}' is not available. Add it to src/models/ and register it in MODELS."
        )

    series = load_series(config["data"], config["seed"])
    train, test = train_test_split(series, config["data"]["test_fraction"])
    predictions = MODELS[model_type](train, test)
    return {"mae": mae(test, predictions), "rmse": rmse(test, predictions), "test_size": int(len(test))}
