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


# please put there paths to your instacne of python enviroments executable
STOCKFORMER_PYTHON_PATH = os.path.join(".venv", "bin", "python")
FINRL_PYTHON_PATH = os.path.join("models", "FinRL", ".venv", "bin", "python")

DATASETS = {
    "nasdaq100":          {"start": "2021-01-01", "end": "2026-01-31"},
    "wig60":              {"start": "2021-01-01", "end": "2026-01-31"},
    "nasdaq100_extended": {"start": "2021-01-01", "end": "2026-01-31"},
    "csi300":             {"start": "2021-01-01", "end": "2026-01-31"},
}

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"\n  ERROR (exit {rc}): {desc}")
        sys.exit(rc)


def download_stockformer(dataset, dates):
    run(
        [STOCKFORMER_PYTHON_PATH, "data/stockformer/download_data.py",
         "--dataset", dataset,
         "--start",   START_DATE, #dates["start"],
         "--end",     END_DATE, #dates["end"]
         ],
        f"Generating data: {dataset} ({dates['start']} → {dates['end']})",
    )


def train_stockformer(dataset):
    run(
        [STOCKFORMER_PYTHON_PATH, "run_experiments.py",
         "--dataset", dataset,
         "--continue_on_error"],
        f"Training: {dataset}",
    )


def download_finrl(dataset, dates):
    run(
        [FINRL_PYTHON_PATH, "models/FinRL/1-data.py",
         "--dataset", dataset],
        f"Generating data: {dataset} ({dates['start']} → {dates['end']})",
    )


def train_finrl(dataset):
    print("=== RUNNING TRAINING ===")
    run(
        [FINRL_PYTHON_PATH, "models/FinRL/2-train.py",
         "--dataset", dataset],
        f"Training: {dataset}",
    )
    print("=== RUNNING BACKTESTING ===")
    run(
        [FINRL_PYTHON_PATH, "models/FinRL/3-backtest.py",
         "--dataset", dataset],
        f"Training: {dataset}",
    )


def main(args, download_fn=download_stockformer, train_fn=train_stockformer):
    datasets = (
        {args.dataset: DATASETS[args.dataset]}
        if args.dataset
        else DATASETS
    )
    for dataset, dates in datasets.items():
        if not args.skip_download:
            download_fn(dataset, dates)
        if not args.skip_train:
            train_fn(dataset)

    print(f"\n{'='*60}")
    print("  All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    models = ['stockformer', 'finrl-baseline']
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='stockformer',
                        choices=models,
                        required=True,
                        help="Choose model to run.")
    parser.add_argument("--dataset", default=None,
                        choices=list(DATASETS.keys()),
                        help="Run for a single dataset only.")
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip data generation.")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training.")
    args = parser.parse_args()
    if args.model == models[0]:
        main(args, download_fn=download_stockformer, train_fn=train_stockformer)
    elif args.model == models[1]:
        main(args, download_fn=download_finrl, train_fn=train_finrl)


