# EXP-003: FedAvg LSTM on Pecan Street homes in Austin, Texas

## Objective

Repeat EXP-002 on homes in a hot climate, where air conditioning drives the load. Every model and federated setting is the same as EXP-002. Nothing was tuned on this data.

## Data

Pecan Street Dataport, free university access, 15-minute file for Austin (25 homes, 2018). Dataset ID `D002` is a placeholder until the Dataset Register has a row.

- Household use = `grid` + `solar` (20 of the 25 homes have solar panels, so `grid` alone goes negative).
- Hourly kWh = mean of the four 15-minute kW readings.
- Period: 1 April to 30 September 2018. The test period is the last quarter, about mid-August to the end of September.
- Same completeness rule as London: gaps of up to 2 hours are filled, and a home with a longer gap is left out.
- **13 of 25 homes are kept.** Ten homes were lost to one recording outage on 20 to 26 July 2018, and two to other gaps. The reasons per home are in `data/processed/pecan_austin/D002_pecan_austin_summary.json`.
- Mean load is 1.63 kWh per hour, against 0.47 for the London homes. Errors here are therefore larger in kWh and cannot be compared with EXP-002 directly.

The licence does not allow sharing the data. It stays in `data/`, which Git ignores.

## Command

```bash
python scripts/prepare_pecan.py --input data/raw/pecan/15minute_data_austin/15minute_data_austin.csv
python scripts/run_experiment.py --config configs/EXP-003_pecan_austin_fedavg_lstm.yaml
```

## Key metrics

Run `20261116-163028`, seed 42, on commit `18faff0` with the new files not yet committed. Test period, mean over the 13 homes, kWh per hour. Lower is better.

| Approach | MAE | RMSE | MAPE (%) |
|---|---|---|---|
| Persistence | 0.4900 | 0.7325 | 34.88 |
| Local-only | 0.4381 | 0.6336 | 33.90 |
| Centralised (not private) | 0.4836 | 0.6727 | 39.20 |
| FedAvg | 0.4533 | 0.6527 | 34.65 |
| FedAvg then fine-tuning | 0.4317 | 0.6255 | 32.99 |

- FedAvg then fine-tuning is the best of the five: RMSE is 1.3% lower and MAE 1.5% lower than local-only.
- Plain FedAvg is worse than local-only (RMSE 3.0% higher, MAE 3.5% higher).
- The order of local-only, FedAvg and fine-tuned FedAvg is the same as in EXP-002 on London homes. The gain from fine-tuning is smaller here (1.3% against 3.9% on RMSE).
- The centralised model is worse than local-only here (RMSE 6.2% higher). On London it had the lowest MAE. It trains for 5 epochs on pooled data, a setting chosen for 100 homes, so it may be under-trained with 13.

## Other seeds

Runs `20261116-170240` (seed 43) and `20261116-170625` (seed 44), started with `--seed`. Test RMSE, kWh per hour.

| Approach | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---|---|---|---|
| Local-only | 0.6336 | 0.6414 | 0.6302 | 0.6351 |
| Centralised (not private) | 0.6727 | 0.6584 | 0.6500 | 0.6603 |
| FedAvg | 0.6527 | 0.6495 | 0.6546 | 0.6523 |
| FedAvg then fine-tuning | 0.6255 | 0.6279 | 0.6260 | 0.6264 |

- Fine-tuned FedAvg beats local-only with all three seeds: RMSE 1.3%, 2.1% and 0.7% lower (mean RMSE 1.4% lower).
- Plain FedAvg is worse than local-only with all three seeds (RMSE 3.0%, 1.3% and 3.9% higher).
- The centralised model is worse than local-only on RMSE with all three seeds, by 2.6% to 6.2%.
- EXP-004 repeats this on 20 homes.

## Limits

- 13 homes and three seeds. The direction is the same with every seed, but the gain is small (0.7% to 2.1% on RMSE).
- Settings come from tuning on London data. 10 of 13 homes train in each round, against 10 of 100 in EXP-002, so the federated setup is not the same in practice.
- Austin is hot, but it is not the Gulf: housing, tariffs and habits differ.
- No weather inputs. Temperature is likely to matter more here than in London.
- Simulated on one laptop; no edge devices, and no communication or energy cost measured.

## Report

No Drive report yet, and the Experiment Log has no row for EXP-003.
