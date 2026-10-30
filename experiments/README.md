# Experiments

One folder per experiment, named with its ID: `EXP-001_short_name/`.

Each run of `scripts/run_experiment.py` creates `runs/<timestamp>/` with:

- `config.yaml`: the exact config used
- `metrics.json`: results
- `run_info.json`: commit hash, date, Python version and seed

Add a short `README.md` per experiment: objective, command, key metrics and a link to the Drive report.
