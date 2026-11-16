"""Try several settings for a federated config and score each on the validation period.

The test period is never used here, so the final test result stays honest.

Usage:
    python scripts/tune.py --config configs/EXP-001_lcl_fedavg_lstm.yaml \
        --set federated.local_epochs=1,2,5 --set model.learning_rate=0.001,0.005

Every combination of the `--set` values is run. Results are printed and saved as a CSV file
under experiments/<id>_<name>/tuning/.
"""
import argparse
import copy
import csv
import itertools
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run  # noqa: E402


def set_value(config: dict, dotted_key: str, value) -> None:
    """Set config["a"]["b"] from the key "a.b"."""
    *parents, last = dotted_key.split(".")
    node = config
    for key in parents:
        node = node.setdefault(key, {})
    node[last] = value


def parse_set(text: str) -> tuple[str, list]:
    """Turn "model.hidden_size=16,32" into ("model.hidden_size", [16, 32])."""
    key, _, values = text.partition("=")
    if not values:
        raise argparse.ArgumentTypeError(f"Expected key=value1,value2 but got '{text}'")
    return key, [yaml.safe_load(v) for v in values.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score settings on the validation period.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", dest="sets", action="append", type=parse_set, default=[], help="key=v1,v2,...")
    parser.add_argument("--validation-fraction", type=float, default=0.15, help="Share of the whole period used to score")
    parser.add_argument(
        "--approaches", default="persistence,fedavg,fedavg_finetuned", help="Comma-separated approaches to run"
    )
    args = parser.parse_args()

    base = yaml.safe_load(args.config.read_text())
    base["data"]["evaluate_on"] = "validation"
    base["data"]["validation_fraction"] = args.validation_fraction
    base["approaches"] = args.approaches.split(",")

    keys = [key for key, _ in args.sets]
    rows = []
    for combination in itertools.product(*[values for _, values in args.sets]):
        config = copy.deepcopy(base)
        for key, value in zip(keys, combination):
            set_value(config, key, value)
        started = time.time()
        metrics = run(config)
        row = dict(zip(keys, combination)) | metrics | {"seconds": round(time.time() - started, 1)}
        rows.append(row)
        shown = ", ".join(f"{k}={v}" for k, v in zip(keys, combination)) or "base settings"
        scores = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k.endswith(("_mae", "_rmse")))
        print(f"{shown}: {scores} ({row['seconds']} s)", flush=True)

    exp = base["experiment"]
    out_dir = ROOT / "experiments" / f"{exp['id']}_{exp['name']}" / "tuning"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.csv"
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
