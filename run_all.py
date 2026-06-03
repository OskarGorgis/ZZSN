"""
run_all.py
Generuje dane i trenuje modele dla wszystkich datasetów.

Użycie:
    python run_all.py                        # generuj + trenuj wszystko
    python run_all.py --skip_download        # tylko trenowanie (dane już są)
    python run_all.py --skip_train           # tylko generowanie danych
    python run_all.py --dataset nasdaq100    # tylko jeden dataset
"""

import argparse
import subprocess
import sys
import os


DATASETS = {
    "nasdaq100":          {"start": "2021-01-01", "end": "2026-01-31"},
    "wig60":              {"start": "2021-01-01", "end": "2026-01-31"},
    "nasdaq100_extended": {"start": "2021-01-01", "end": "2026-01-31"},
}

# Okres, na którym model jest TESTOWANY (trening zawsze wcześniejszy)
TEST_START = "2023-01-01"
TEST_END   = "2025-12-31"


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}")
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"\n  BLAD (exit {rc}): {desc}")
        sys.exit(rc)


def download(dataset, dates):
    run(
        [sys.executable, "data/stockformer/download_data.py",
         "--dataset",    dataset,
         "--start",      dates["start"],
         "--end",        dates["end"],
         "--test_start", TEST_START,
         "--test_end",   TEST_END],
        f"Generowanie danych: {dataset} ({dates['start']} → {dates['end']})"
        f"  test: {TEST_START} → {TEST_END}",
    )


def train(dataset):
    run(
        [sys.executable, "run_experiments.py",
         "--dataset", dataset,
         "--continue_on_error"],
        f"Trenowanie: {dataset}",
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
    print("  Wszystko gotowe!")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        choices=list(DATASETS.keys()),
                        help="Uruchom tylko dla jednego datasetu.")
    parser.add_argument("--skip_download", action="store_true",
                        help="Pomiń generowanie danych.")
    parser.add_argument("--skip_train", action="store_true",
                        help="Pomiń trenowanie.")
    args = parser.parse_args()
    main(args)
