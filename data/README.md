# Data

Data is **not** stored in Git. Keep it in Drive (`03_Datasets`) and copy it here locally.

- `raw/`: original files, never edited.
- `processed/`: cleaned files produced by code in `src/data/`.

Name files with the Dataset ID from the Dataset Register, for example `D001_deddiag_home01.csv`.

## Low Carbon London (D001)

Smart meter readings for 5,567 London households, 2011 to 2014, every 30 minutes (UK Power Networks, CC BY 4.0).

1. Download `Partitioned LCL Data.zip` (about 760 MB) from the [London Datastore](https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households) and save it as `data/raw/lcl/Partitioned_LCL_Data.zip`.
2. Run `python scripts/prepare_lcl.py --input data/raw/lcl/Partitioned_LCL_Data.zip`.

This writes one hourly file per household to `data/processed/lcl/` (columns `time`, `load_kwh`) and a `D001_lcl_summary.json` that lists the chosen households. The defaults follow Briggs, Fan and Andras (2022): 100 flat-rate homes, January to June 2013. Homes with gaps longer than two hours are left out.

## Pecan Street Dataport, Austin (D002)

Circuit-level power for 25 homes in Austin, Texas, for 2018, every 15 minutes (Pecan Street Inc.). Access needs a free university account, and the licence does not allow sharing the data, so do not put it in Drive or Git.

1. Log in at [dataport.pecanstreet.org/academic](https://dataport.pecanstreet.org/academic), download `Austin 15-min` (22.4 MB) and unpack it into `data/raw/pecan/`.
2. Run `python scripts/prepare_pecan.py --input data/raw/pecan/15minute_data_austin/15minute_data_austin.csv`.

This writes one hourly file per household to `data/processed/pecan_austin/` (columns `time`, `load_kwh`) and a `D002_pecan_austin_summary.json`. Household use is `grid` + `solar`. The default period is April to September 2018. Homes with gaps longer than two hours are left out, which keeps 13 of the 25. Adding `--week-fill-hours 24` fills gaps of up to 24 hours from the week before and keeps 20 (used in EXP-004).

The same page has 15-minute files for California (23 homes, one year each between 2014 and 2018), New York (25 homes, May to October 2019) and Puerto Rico (25 homes, September to November 2023). The script reads them with `--input`, `--start`, `--end`, `--out` and `--prefix`; for Puerto Rico add `--pattern "pr_realpower_*"`.
