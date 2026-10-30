# Instructions for Claude Code

Project: Edge Computing-Based Federated Learning for Energy Consumption Prediction and Optimisation.
Follow CONTRIBUTING.md for branches, commits and experiments.

## When a team member reports an experiment

1. Use the Experiment ID they give (for example `EXP-003`). If none is given, ask for it.
2. Work on branch `exp/EXP-xxx-short-name`; create it from `main` if needed.
3. Save or update the config in `configs/EXP-xxx_short_name.yaml`.
4. Save results to `experiments/EXP-xxx_short_name/runs/<timestamp>/` with `metrics.json` and the config used.
5. Add a short `experiments/EXP-xxx_short_name/README.md`: objective, command, key metrics, link to the Drive report.
6. Commit with `exp: <what was run and the main result>`, push the branch and open a pull request.

## Rules

- Never commit to `main` directly. Never force-push.
- Never commit files from `data/`, model weights (`*.pt`, `*.h5`, `*.pkl`) or secrets.
- Report numbers exactly as measured. Do not round away differences or invent results.
- Keep reusable code in `src/`; keep notebooks for exploration only.
- Run `pytest` before pushing.
