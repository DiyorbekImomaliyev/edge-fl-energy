# Edge FL Energy

Code for the research project **Edge Computing-Based Federated Learning for Energy Consumption Prediction and Optimisation**.

Documents, papers and reports live in the team's Google Drive: start at the [Research Hub](https://docs.google.com/document/d/1UWLMx2JTn_Ugsc-eh4J6_FRdriEQCVmZDHd0ZU4DGZo/edit).
This repository holds code, configs and experiment results.

## Quick start

Requires Python 3.11 or newer and Git.

```bash
git clone https://github.com/<org-or-user>/edge-fl-energy.git
cd edge-fl-energy
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_experiment.py --config configs/EXP-000_persistence_baseline.yaml
pytest
```

The example run trains nothing. It forecasts a synthetic load series with a persistence baseline and saves a full run record to `experiments/EXP-000_persistence_baseline/runs/`.

To try federated training, run the second example:

```bash
python scripts/run_experiment.py --config configs/EXP-000_fedavg_lstm_example.yaml
```

It trains a small LSTM on generated homes and reports the same metrics for five approaches: persistence, local-only training, centralised training, FedAvg, and FedAvg followed by local fine-tuning. The data is generated, so its numbers only show that the code works.

To run on real homes, prepare the Low Carbon London data as described in [data/README.md](data/README.md), then run `configs/EXP-001_lcl_fedavg_lstm.yaml`.

## Repository structure

```
configs/        One YAML file per experiment (EXP-001_short_name.yaml)
data/           Local data only. Not tracked by Git; see data/README.md
experiments/    Run records (metrics, config, commit) per experiment
notebooks/      Exploration only; move finished code into src/
scripts/        Entry points, such as run_experiment.py
src/
  data/         Loading and preprocessing
  models/       Forecasting models (baselines, LSTM, GRU, XGBoost)
  federated/    FedAvg and the comparison against its baselines
  optimisation/ Energy and comfort optimisation (for example PSO)
  evaluation/   Metrics
  pipeline.py   Connects data, model and metrics for one run
tests/          Automated tests (run with pytest)
```

## How we work

Read [CONTRIBUTING.md](CONTRIBUTING.md) before your first change. In short:

1. Pick or create a task in the Task Tracker and a GitHub issue.
2. Work on a branch, never directly on `main`.
3. Open a pull request; one teammate reviews and merges it.
4. Log every experiment in the Drive Experiment Log with the same `EXP` ID.
