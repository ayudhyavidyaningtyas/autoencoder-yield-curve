#!/usr/bin/env python3
"""Run the yield-curve autoencoder replication on the supplied US dataset."""

from __future__ import annotations

import sys
from pathlib import Path

import replicate_autoencoder_yield_curve as core


ROOT = Path(__file__).resolve().parents[1]
PROJECT_US_CSV = ROOT / "data" / "USdataYC.csv"


def main() -> None:
    default_args = [
        "--csv",
        str(PROJECT_US_CSV),
        "--csv-format",
        "generic",
        "--date-format",
        "%m/%d/%y",
        "--dataset-name",
        "US Treasury",
        "--maturities",
        "2Y,3Y,5Y,7Y,10Y,20Y",
        "--start-date",
        "1993-10-01",
        "--end-date",
        "2023-12-29",
        "--output-dir",
        str(ROOT / "outputs_us"),
    ]
    sys.argv = [sys.argv[0], *default_args, *sys.argv[1:]]
    core.main()


if __name__ == "__main__":
    main()
