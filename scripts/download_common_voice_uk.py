#!/usr/bin/env python3
"""Download the official Ukrainian Common Voice release from Mozilla Data Collective."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DATASET_ID = "cmqinqaqp00wunq07w9oyei38"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("training/uk_UA/raw"))
    parser.add_argument("--dataset-id", default=DATASET_ID)
    args = parser.parse_args()

    if not os.environ.get("MDC_API_KEY"):
        raise SystemExit(
            "MDC_API_KEY is missing. Create an API key at "
            "https://mozilladatacollective.com/api-reference and export it first."
        )
    try:
        from datacollective import download_dataset
    except ImportError as error:
        raise SystemExit(
            "Install the official client with: "
            ".venv-nix/bin/pip install datacollective"
        ) from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = download_dataset(
        args.dataset_id,
        download_directory=str(args.output_dir),
        enable_logging=True,
    )
    print(f"Downloaded Common Voice dataset to {path}")


if __name__ == "__main__":
    main()
