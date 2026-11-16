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
