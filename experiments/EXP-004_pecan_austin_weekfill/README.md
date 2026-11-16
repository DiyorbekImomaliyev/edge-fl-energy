# EXP-004: FedAvg LSTM on 20 Pecan Street homes in Austin, three seeds

## Objective

Repeat EXP-003 with more homes and with three seeds, to see whether its small gain from federated learning holds.

## Data

Same source, period and rules as EXP-003 (Pecan Street Dataport, Austin, 1 April to 30 September 2018, hourly), with one change: gaps of 3 to 24 hours are filled with the same hours one week earlier. This brings back seven homes that lost 11 to 24 hours in the recording outage of 20 to 26 July 2018, or in May.

- **20 of 25 homes are kept.** The other five have gaps of 72 to 616 hours and are left out.
- Every filled gap is before the test period, which starts on 16 August 2018. Test scores use measured readings only.
- The filled hours are copies, not measurements. They are at most 24 of the 3,294 training hours of a home.
- Dataset ID `D002` is a placeholder. The licence does not allow sharing the data; it stays in `data/`.

## Command

```bash
python scripts/prepare_pecan.py --input data/raw/pecan/15minute_data_austin/15minute_data_austin.csv --week-fill-hours 24 --out data/processed/pecan_austin_weekfill
python scripts/run_experiment.py --config configs/EXP-004_pecan_austin_weekfill.yaml --seed 42
python scripts/run_experiment.py --config configs/EXP-004_pecan_austin_weekfill.yaml --seed 43
python scripts/run_experiment.py --config configs/EXP-004_pecan_austin_weekfill.yaml --seed 44
```

## Key metrics

Runs `20261116-170306` (seed 42), `20261116-170708` (seed 43) and `20261116-171044` (seed 44), on commit `18faff0` with the new files not yet committed. Settings are those of EXP-002; nothing was tuned on this data. Test period, mean over the 20 homes, kWh per hour. Lower is better.

RMSE:

| Approach | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---|---|---|---|
| Persistence | 0.7871 | 0.7871 | 0.7871 | 0.7871 |
| Local-only | 0.6764 | 0.6859 | 0.6828 | 0.6817 |
| Centralised (not private) | 0.6922 | 0.6983 | 0.6960 | 0.6955 |
| FedAvg | 0.7063 | 0.6960 | 0.7002 | 0.7008 |
| FedAvg then fine-tuning | 0.6653 | 0.6653 | 0.6657 | 0.6654 |

MAE:

| Approach | Seed 42 | Seed 43 | Seed 44 | Mean |
|---|---|---|---|---|
| Persistence | 0.5293 | 0.5293 | 0.5293 | 0.5293 |
| Local-only | 0.4786 | 0.4856 | 0.4860 | 0.4834 |
| Centralised (not private) | 0.4885 | 0.4835 | 0.5114 | 0.4945 |
| FedAvg | 0.5013 | 0.4927 | 0.5011 | 0.4984 |
| FedAvg then fine-tuning | 0.4678 | 0.4689 | 0.4664 | 0.4677 |

- FedAvg then fine-tuning is the best approach with every seed. Against local-only its RMSE is 1.7%, 3.0% and 2.5% lower (mean RMSE 2.4% lower) and its MAE 2.3%, 3.4% and 4.0% lower (mean MAE 3.2% lower).
- Its RMSE hardly moves between seeds (0.6653 to 0.6657), while local-only moves from 0.6764 to 0.6859. The gap between the two is larger than that spread.
- Plain FedAvg is worse than local-only with every seed (RMSE 1.5% to 4.4% higher).
- The centralised model is worse than local-only on RMSE with every seed (1.8% to 2.3% higher).
- The gain over local-only is larger than with 13 homes in EXP-003 (mean RMSE 2.4% lower against 1.4%). The two home sets differ, so this does not show that more homes cause the larger gain.

## Limits

- 20 homes from one city and one summer. Three seeds show the result is not seed noise; they do not show it holds for other homes.
- Settings come from tuning on London data. 10 of 20 homes train in each round, against 10 of 100 in EXP-002.
- Errors cannot be compared in kWh with the London runs: these homes use about four times as much electricity (1.83 kWh per hour against 0.47).
- Austin is hot, but it is not the Gulf: housing, tariffs and habits differ.
- No weather inputs. Simulated on one laptop; no edge devices, and no communication or energy cost measured.

## Report

No Drive report yet, and the Experiment Log has no row for EXP-004.
