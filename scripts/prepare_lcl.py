"""Turn the raw Low Carbon London download into one hourly CSV file per household.

Usage:
    python scripts/prepare_lcl.py --input "data/raw/lcl/Partitioned_LCL_Data.zip"

Download the raw data first (about 760 MB, CC BY 4.0) from
https://data.london.gov.uk/dataset/smartmeter-energy-use-data-in-london-households
and keep it in data/raw/, which Git ignores.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lcl import choose_homes, hourly_homes, read_raw, write_client_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Low Carbon London data for federated runs.")
    parser.add_argument("--input", required=True, type=Path, help="Raw zip file, folder of CSV files, or one CSV file")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/lcl", help="Folder for the client files")
    parser.add_argument("--start", default="2013-01-01", help="First day (included)")
    parser.add_argument("--end", default="2013-07-01", help="Last day (not included)")
    parser.add_argument("--homes", type=int, default=100, help="Number of households to keep")
    parser.add_argument("--tariff", default="Std", help="Std (flat rate), ToU, or 'all'")
    parser.add_argument("--max-gap-hours", type=int, default=2, help="Longest gap that is filled by interpolation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for choosing the households")
    parser.add_argument("--prefix", default="D001_lcl", help="File name prefix (Dataset ID from the Dataset Register)")
    args = parser.parse_args()

    tariff = None if args.tariff == "all" else args.tariff
    frame = read_raw(args.input, args.start, args.end)
    homes = hourly_homes(frame, args.start, args.end, tariff, args.max_gap_hours)
    names = choose_homes(homes, args.homes, args.seed)
    paths = write_client_files(homes, names, args.out, args.prefix)

    summary = {
        "source": str(args.input),
        "start": args.start,
        "end": args.end,
        "tariff": args.tariff,
        "max_gap_hours": args.max_gap_hours,
        "seed": args.seed,
        "raw_rows_in_period": int(len(frame)),
        "homes_in_period": int(frame["home"].nunique()),
        "complete_homes_on_tariff": len(homes),
        "homes_written": len(paths),
        "hours_per_home": int(len(homes[names[0]])),
        "homes": names,
    }
    (args.out / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Wrote {len(paths)} homes with {summary['hours_per_home']} hours each to {args.out} "
        f"({len(homes)} complete {args.tariff} homes were available)."
    )


if __name__ == "__main__":
    main()
