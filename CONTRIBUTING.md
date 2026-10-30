# Contributing

## Set up once

Follow the Quick start in [README.md](README.md). Use your own branch for every change.

## Daily workflow

1. Update `main`: `git checkout main && git pull`
2. Create a branch: `git checkout -b <type>/<ID>-short-name`
3. Commit small, clear changes.
4. Push and open a pull request: `git push -u origin <branch>`
5. Ask one teammate to review. Merge after approval and passing tests.
6. Delete the branch after merging.

## Branch names

| Type | Use for | Example |
|---|---|---|
| `exp/` | An experiment | `exp/EXP-001-fedavg-lstm` |
| `feat/` | New code | `feat/T004-deddiag-loader` |
| `fix/` | Bug fix | `fix/T009-wrong-split` |
| `docs/` | Documentation | `docs/T002-readme-setup` |

Use the Task ID (`T001`) or Experiment ID (`EXP-001`) from the Drive trackers.

## Commit messages

Format: `<type>: <what changed>`, in the imperative, under 72 characters.

```
feat: add DEDDIAG loader for one home
exp: run EXP-001 FedAvg LSTM with 5 clients
fix: correct train/test split order
docs: add setup steps for Windows
```

Types: `feat`, `fix`, `exp`, `docs`, `test`, `refactor`, `chore`.

## Run an experiment

1. Add a row to the Experiment Log in Drive and take the next ID, for example `EXP-001`.
2. Create a branch: `exp/EXP-001-short-name`.
3. Copy `configs/EXP-000_persistence_baseline.yaml` to `configs/EXP-001_short_name.yaml` and edit it.
4. Run: `python scripts/run_experiment.py --config configs/EXP-001_short_name.yaml`
5. Check the run record in `experiments/EXP-001_short_name/runs/<timestamp>/`.
6. Write the report in Drive from the Experiment Report template.
7. Commit the config and run record, open a pull request, and add the commit hash to the Experiment Log.

Commit your code before a final run. The run record stores the commit hash, so anyone can reproduce the result.

## Rules

- Never commit raw or processed data, model weights or passwords. Data lives in Drive; see [data/README.md](data/README.md).
- Clear notebook outputs before committing.
- Keep one task per pull request.
- Run `pytest` before opening a pull request.
- Ask before changing shared code in `src/` that other experiments use.
