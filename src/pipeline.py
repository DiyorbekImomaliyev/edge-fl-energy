"""Run one experiment: load data, forecast, evaluate."""
from src.data.loaders import load_series, train_test_split
from src.evaluation.metrics import mae, rmse
from src.models.baselines import persistence_forecast

# Add new models here: "name": function(train, test) -> predictions
MODELS = {"persistence": persistence_forecast}


def run(config: dict) -> dict:
    """Run the experiment described by `config` and return its metrics."""
    model_type = config["model"]["type"]
    if model_type not in MODELS:
        raise NotImplementedError(
            f"Model '{model_type}' is not available. Add it to src/models/ and register it in MODELS."
        )
    if config.get("federated", {}).get("enabled"):
        raise NotImplementedError("Federated training is not implemented yet. Add it in src/federated/.")

    series = load_series(config["data"], config["seed"])
    train, test = train_test_split(series, config["data"]["test_fraction"])
    predictions = MODELS[model_type](train, test)
    return {"mae": mae(test, predictions), "rmse": rmse(test, predictions), "test_size": int(len(test))}
