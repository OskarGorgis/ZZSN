"""
run_experiments.py
Runs Stockformer training for every rolling window of a dataset.

Usage:
    python run_experiments.py --dataset nasdaq100
    python run_experiments.py --dataset smoke_test --window 0
    python run_experiments.py --dataset nasdaq100 --continue_on_error
"""

import argparse
import os
import subprocess
import sys


def find_configs(config_dir, dataset):
    d = os.path.join(config_dir, dataset)
    if not os.path.isdir(d):
        print(f"Config directory not found: {d}")
        print("Run download_data.py first to generate data and configs.")
        sys.exit(1)
    configs = sorted(
        os.path.abspath(os.path.join(d, f))
        for f in os.listdir(d)
        if f.endswith(".conf")
    )
    if not configs:
        print(f"No .conf files found in {d}")
        sys.exit(1)
    return configs


def run_window(config_path, stockformer_dir):
    cmd = [sys.executable, "MultiTask_Stockformer_train.py", "--config", config_path]
    return subprocess.run(cmd, cwd=stockformer_dir).returncode


def main(args):
    stockformer_dir = os.path.join(os.path.dirname(__file__), "models/stockformer")
    configs = find_configs(args.config_dir, args.dataset)

    if args.window is not None:
        if args.window >= len(configs):
            print(f"Window index {args.window} out of range (0–{len(configs)-1})")
            sys.exit(1)
        configs = [configs[args.window]]

    print(f"Dataset : {args.dataset}")
    print(f"Windows : {len(configs)}")
    print(f"Config  : {os.path.dirname(configs[0])}\n")

    failed = []
    for i, cfg in enumerate(configs):
        label = os.path.splitext(os.path.basename(cfg))[0].replace("window_", "")
        print(f"{'='*60}")
        print(f"[{i+1}/{len(configs)}] {label}")
        print(f"{'='*60}")
        rc = run_window(cfg, stockformer_dir)
        if rc != 0:
            failed.append(label)
            print(f"  FAILED (exit {rc})")
            if not args.continue_on_error:
                sys.exit(rc)

    print(f"\n{'='*60}")
    if failed:
        print(f"Finished with {len(failed)} failure(s): {', '.join(failed)}")
    else:
        print(f"All {len(configs)} window(s) completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="smoke_test",
                        choices=["smoke_test", "nasdaq100", "wig60", "nasdaq100_extended", "csi300"])
    parser.add_argument("--config_dir", default="models/stockformer/config")
    parser.add_argument("--window", type=int, default=None,
                        help="Run only this window index (0-based). Omit to run all.")
    parser.add_argument("--continue_on_error", action="store_true",
                        help="Keep going if a window fails instead of stopping.")
    args = parser.parse_args()
    main(args)
