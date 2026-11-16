"""Turn a raw Pecan Street Dataport 15-minute download into one hourly CSV file per household.

Usage:
    python scripts/prepare_pecan.py --input data/raw/pecan/15minute_data_austin/15minute_data_austin.csv

Download the raw data first from https://dataport.pecanstreet.org/academic (free university
account needed) and keep it in data/raw/, which Git ignores. The licence does not allow
sharing the data, so never commit it.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.lcl import write_client_files  # noqa: E402
from src.data.pecan import hourly_homes, read_raw  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Pecan Street data for federated runs.")
    parser.add_argument("--input", required=True, type=Path, help="Raw 15-minute CSV file, or a folder of them")
    parser.add_argument("--pattern", default="*.csv*", help="File name pattern when --input is a folder")
    parser.add_argument("--out", type=Path, default=ROOT / "data/processed/pecan_austin", help="Folder for the client files")
    parser.add_argument("--start", default="2018-04-01", help="First day (included)")
    parser.add_argument("--end", default="2018-10-01", help="Last day (not included)")
    parser.add_argument("--max-gap-hours", type=int, default=2, help="Longest gap that is filled by interpolation")
    parser.add_argument("--max-negative-share", type=float, default=0.01, help="Leave out homes with more negative readings than this")
    parser.add_argument("--week-fill-hours", type=int, default=0, help="Fill longer gaps up to this many hours from the week before (0 = off)")
    parser.add_argument("--prefix", default="D002_pecan_austin", help="File name prefix (Dataset ID from the Dataset Register)")
    args = parser.parse_args()

    frame = read_raw(args.input, args.pattern)
    homes, dropped = hourly_homes(frame, args.start, args.end, args.max_gap_hours, args.max_negative_share, args.week_fill_hours)
    if not homes:
        raise SystemExit(f"No complete homes for {args.start} to {args.end}. Left out: {dropped}")
    names = sorted(homes)
    paths = write_client_files(homes, names, args.out, args.prefix)

    summary = {
        "source": str(args.input),
        "start": args.start,
        "end": args.end,
        "max_gap_hours": args.max_gap_hours,
        "max_negative_share": args.max_negative_share,
        "week_fill_hours": args.week_fill_hours,
        "homes_in_file": int(frame["home"].nunique()),
        "homes_written": len(paths),
        "hours_per_home": int(len(homes[names[0]])),
        "homes": names,
        "homes_left_out": dropped,
    }
    (args.out / f"{args.prefix}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(paths)} homes with {summary['hours_per_home']} hours each to {args.out}.")
    for home, reason in dropped.items():
        print(f"Left out home {home}: {reason}")


if __name__ == "__main__":
    main()
