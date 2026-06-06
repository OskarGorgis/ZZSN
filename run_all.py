"""
run_all.py
Generates data and trains models for all datasets.

Usage:
    python run_all.py                        # download + train everything
    python run_all.py --skip_download        # training only (data exists)
    python run_all.py --skip_train           # data generation only
    python run_all.py --dataset nasdaq100    # single dataset only
"""

import argparse
import subprocess
import sys
import os


DATASETS = {
    "nasdaq100":          {"start": "2021-01-01", "end": "2026-01-31"},
    "wig60":              {"start": "2021-01-01", "end": "2026-01-31"},
    "nasdaq100_extended": {"start": "2021-01-01", "end": "2026-01-31"},
    "csi300":             {"start": "2021-01-01", "end": "2026-01-31"},
}


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"\n  ERROR (exit {rc}): {desc}")
        sys.exit(rc)


def download(dataset, dates):
    run(
        [sys.executable, "data/stockformer/download_data.py",
         "--dataset", dataset,
         "--start",   dates["start"],
         "--end",     dates["end"]],
        f"Generating data: {dataset} ({dates['start']} → {dates['end']})",
    )


def train(dataset):
    run(
        [sys.executable, "run_experiments.py",
         "--dataset", dataset,
         "--continue_on_error"],
        f"Training: {dataset}",
    )


def main(args):
    datasets = (
        {args.dataset: DATASETS[args.dataset]}
        if args.dataset
        else DATASETS
    )

    for dataset, dates in datasets.items():
        if not args.skip_download:
            download(dataset, dates)
        if not args.skip_train:
            train(dataset)

    print(f"\n{'='*60}")
    print("  All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        choices=list(DATASETS.keys()),
                        help="Run for a single dataset only.")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip data generation.")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training.")
    args = parser.parse_args()
    main(args)
