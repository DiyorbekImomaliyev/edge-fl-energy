# EXP-001: FedAvg LSTM on Low Carbon London

## Objective

Check on real homes whether federated training beats each home training alone and beats simply copying the last reading, one hour ahead.

## Setup

- Data: D001, Low Carbon London. 100 flat-rate homes chosen at random (seed 42), 1 January to 30 June 2013, hourly kWh. The last 25% of the period is the test set (1,086 hours per home).
- Model: LSTM, 1 layer of 32 units, 24 past hours in, 1 hour ahead. Each home scales its data with its own minimum and maximum.
- FedAvg: 30 rounds, 10 of 100 homes per round, 2 local epochs. Fine-tuning: 3 local epochs.
- Baselines: persistence, local-only (20 epochs per home), centralised (5 epochs on all homes' data).

## Command

```bash
python scripts/prepare_lcl.py --input data/raw/lcl/Partitioned_LCL_Data.zip
python scripts/run_experiment.py --config configs/EXP-001_lcl_fedavg_lstm.yaml
```

## Key metrics

Run `20261116-091724`, commit `bb3218a`, seed 42. Mean over the 100 homes, in kWh per hour. Lower is better.

| Approach | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Persistence | 0.1785 | 0.3352 | 66.90 |
| Local-only | 0.1732 | 0.2699 | 92.06 |
| Centralised (not private) | 0.1545 | 0.2746 | 63.81 |
| FedAvg | 0.1769 | 0.2819 | 95.74 |
| FedAvg then fine-tuning | 0.1639 | 0.2646 | 80.26 |

- Plain FedAvg is worse than local-only training: RMSE is 4.5% higher and MAE 2.1% higher.
- FedAvg then fine-tuning is better than local-only: RMSE is 1.9% lower and MAE 5.4% lower. It has the lowest RMSE of all five.
- The centralised model has the lowest MAE, but it needs every home to share its data.
- Every trained model beats persistence on RMSE. On MAE, plain FedAvg is only 0.9% better than persistence.

## Limits

- One run with one seed, so small differences may not hold with another seed.
- MAPE is not reliable here. Many homes use close to zero in some hours, which makes percentage errors very large. 14 of the 100 homes have hours with exactly zero use, and those hours are left out of MAPE.
- No weather or calendar inputs. Settings were not tuned.
- Simulated on one laptop. No edge devices and no communication cost measured.
- Briggs, Fan and Andras (2022) report RMSE on values scaled with one global minimum and maximum, so their numbers cannot be compared directly with this table.

## Report

No Drive report yet, and the Experiment Log has no row for EXP-001 yet.
