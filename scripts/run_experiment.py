"""Run one experiment from a config file and save a reproducible run record.

Usage:
    python scripts/run_experiment.py --config configs/EXP-000_persistence_baseline.yaml
"""
import argparse
import json
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import run  # noqa: E402


def git(*args: str) -> str:
    """Return the output of a git command, or 'unknown' outside a repository."""
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def set_seed(seed: int) -> None:
    """Fix random seeds so runs can be repeated."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment from a config file.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the experiment YAML file")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    exp = config["experiment"]
    status = git("status", "--porcelain")
    run_info = {
        "experiment_id": exp["id"],
        "owner": exp.get("owner"),
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"),
        "uncommitted_changes": None if status == "unknown" else bool(status),
        "python": platform.python_version(),
        "seed": config["seed"],
    }

    set_seed(config["seed"])
    metrics = run(config)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / "experiments" / f"{exp['id']}_{exp['name']}" / "runs" / stamp
    run_dir.mkdir(parents=True)
    shutil.copy(args.config, run_dir / "config.yaml")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2) + "\n")

    summary = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    print(f"{exp['id']} finished: {summary}")
    print(f"Run record: {run_dir.relative_to(ROOT)}")
    if run_info["uncommitted_changes"]:
        print("Warning: uncommitted changes. Commit your code before a final run.")


if __name__ == "__main__":
    main()
