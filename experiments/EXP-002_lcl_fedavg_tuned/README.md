# EXP-002: FedAvg LSTM on Low Carbon London, tuned

## Objective

Repeat EXP-001 with the settings that tuning supported, and score it once on the test period, which tuning never used.

## What was tuned, and how

Settings were chosen on a validation period: the last 15% of the whole period before the test period starts. The test period (the last 25%) was cut off and not read during tuning (`scripts/tune.py`, `data.evaluate_on: validation`). Tuning files are in `experiments/EXP-001_lcl_fedavg_lstm/tuning/`.

Validation RMSE of FedAvg then fine-tuning, 100 homes, kWh per hour, seed 42. Lower is better.

| Changed setting | Values tried | Validation RMSE | Kept? |
|---|---|---|---|
| Local epochs (30 rounds) | 1 / 2 / 5 | 0.2790 / 0.2715 / 0.2656 | 5 |
| Calendar inputs (hour, weekday), 5 local epochs | off / on | 0.2656 / 0.2873 | off |
| Rounds, 5 local epochs | 30 / 60 | 0.2656 / 0.2670 | 30 |
| Learning rate, 60 rounds | 0.005 / 0.01 | 0.2670 / 0.2669 | 0.005 |
| Clustered FedAvg, then fine-tuning | 2 / 4 / 8 groups | 0.2675 / 0.2711 / 0.2674 | not used |

So the only change from EXP-001 is 5 local epochs instead of 2.

## Command

```bash
python scripts/prepare_lcl.py --input data/raw/lcl/Partitioned_LCL_Data.zip
python scripts/run_experiment.py --config configs/EXP-002_lcl_fedavg_tuned.yaml
```

## Key metrics

Run `20261116-103606`, commit `138fc64`, seed 42. Test period, mean over the 100 homes, kWh per hour. Lower is better.

| Approach | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Persistence | 0.1785 | 0.3352 | 66.90 |
| Local-only | 0.1749 | 0.2706 | 92.31 |
| Centralised (not private) | 0.1551 | 0.2731 | 66.74 |
| FedAvg | 0.1743 | 0.2811 | 94.55 |
| FedAvg then fine-tuning | 0.1620 | 0.2602 | 82.19 |

- FedAvg then fine-tuning is better than local-only: RMSE is 3.9% lower and MAE 7.3% lower. It has the lowest RMSE of the five.
- Plain FedAvg is still worse than local-only on RMSE (3.9% higher), and about equal on MAE (0.3% lower).
- The centralised model has the lowest MAE, but needs every home to share its data.
- Against EXP-001, fine-tuned FedAvg improved from RMSE 0.2646 to 0.2602 (1.7% lower).

## What did not help

- **Calendar inputs** made every validation result worse. I did not find out why. Their values run from -1 to 1 while the scaled load is mostly near 0, so they may swamp the load early in training, or there may be a fault in how they are built. This needs checking before anyone concludes that time of day does not matter.
- **More rounds** (60) made plain FedAvg worse on validation (RMSE 0.2830 to 0.2925), which suggests the shared model moves around from round to round with 10 of 100 homes per round.
- **Clustering** by model updates did not beat plain fine-tuning at 2, 4 or 8 groups. Briggs, Fan and Andras (2022) found it gave the same accuracy for less computation; computation was not measured here.

## Limits

- One seed. The gaps between approaches are a few percent, so they may not hold with another seed. The local-only numbers differ slightly from EXP-001 (RMSE 0.2706 against 0.2699) although its settings are the same, because all approaches draw from one random stream and FedAvg used more of it here. That difference is a measure of the seed noise.
- MAPE is not reliable on this data: many hours are close to zero.
- No weather inputs. Simulated on one laptop; no edge devices, and no communication or energy cost measured.
- London homes in a mild climate, not a hot one.

## Report

No Drive report yet, and the Experiment Log has no row for EXP-002.
