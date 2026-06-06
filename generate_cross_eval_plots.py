"""
generate_cross_eval_plots.py
Generates cross-evaluation plots from existing CSV results.
Does NOT re-run model inference — reads saved CSVs + training logs.

Usage:
    python generate_cross_eval_plots.py
    python generate_cross_eval_plots.py --model_label "Stockformer v2"
"""
from __future__ import annotations

import csv
import argparse
from pathlib import Path

from src.log_parser import load_dataset
from src import plots

ROOT     = Path(__file__).parent
CSV_DIR  = ROOT / "data" / "results" / "cross_eval"
OUT_DIR  = ROOT / "data" / "results2" / "cross_eval"
DATASETS = ["nasdaq100", "wig60", "nasdaq100_extended"]


def load_all_cross() -> dict:
    """Return {(source, target): [rows]} from all *_to_*.csv files."""
    data = {}
    for path in sorted(CSV_DIR.glob("*_to_*.csv")):
        rows = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "source":   row["source_dataset"],
                    "target":   row["target_dataset"],
                    "window":   row["target_window"],
                    "cls_acc":  float(row["cls_acc"]),
                    "reg_mae":  float(row["reg_mae"]),
                    "reg_rmse": float(row["reg_rmse"]),
                })
        if rows:
            key = (rows[0]["source"], rows[0]["target"])
            data[key] = rows
    return data


def load_indomain() -> dict:
    """Return {dataset: {window: {cls_acc, reg_mae, reg_rmse}}} from training logs."""
    result = {}
    for ds in DATASETS:
        logs = load_dataset(ds)
        result[ds] = {
            w: d["test"]
            for w, d in logs.items()
            if d.get("test")
        }
    return result


def _as_cross_plot_format(cross_raw: dict) -> dict:
    """Convert {(src, tgt): [rows]} → format expected by src/plots.py functions."""
    return cross_raw  # rows already have the right keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_label", default="Stockformer",
                        help="Label used in plot titles.")
    args = parser.parse_args()

    print("Loading cross-eval CSVs...")
    cross_data = load_all_cross()
    if not cross_data:
        print("No CSV files found in data/results/cross_eval/")
        print("Run  python cross_eval.py  first.")
        return

    short = {"nasdaq100": "NASDAQ", "wig60": "WIG60",
             "nasdaq100_extended": "NASDAQ+"}
    pairs_str = ", ".join(f"{short.get(s,s)}→{short.get(t,t)}"
                          for s, t in sorted(cross_data))
    print(f"  Found {len(cross_data)} pairs: {pairs_str}")

    print("\nLoading in-domain metrics from training logs...")
    indomain = load_indomain()
    for ds, wins in indomain.items():
        print(f"  {ds}: {len(wins)} windows")

    print("\nGenerating plots...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plots.plot_cross_eval_overview(cross_data, indomain, OUT_DIR,
                                   args.model_label)

    plots.plot_transfer_delta(cross_data, indomain, OUT_DIR)

    plots.plot_cross_eval_temporal(cross_data, indomain, OUT_DIR)

    # build all_rows including in-domain diagonal for the heatmap
    all_rows = []
    for ds, wins in indomain.items():
        for w, m in wins.items():
            all_rows.append({"source_dataset": ds, "target_dataset": ds, **m})
    for (src, tgt), rows in cross_data.items():
        for r in rows:
            all_rows.append({
                "source_dataset": src,
                "target_dataset": tgt,
                "cls_acc":  r["cls_acc"],
                "reg_mae":  r["reg_mae"],
                "reg_rmse": r["reg_rmse"],
            })
    plots.plot_heatmaps(all_rows, DATASETS, OUT_DIR)

    print(f"\nDone.  Plots saved to:  data/results/cross_eval/")
    print("  cross_eval_overview.png  — main grouped comparison by target dataset")
    print("  transfer_delta.png       — Δ (cross − in-domain) per pair")
    print("  cross_eval_temporal.png  — transfer quality over time")
    print("  heatmap_*.png            — source × target matrix")


if __name__ == "__main__":
    main()
