"""
cross_eval.py
Evaluate a model trained on one dataset (source) on test data from another (target).
Demonstrates zero-shot cross-market transfer.

Architecture note: all Stockformer weight matrices have shape [F, F] — the number
of stocks N is just a batch dimension, so a checkpoint from one market works
directly on another without any weight changes.

Usage:
    python cross_eval.py                              # all cross-dataset pairs
    python cross_eval.py --source nasdaq100 --target wig60
    python cross_eval.py --source nasdaq100 --target wig60 --gpu 0
"""
from __future__ import annotations

import sys
import csv
import math
import argparse
import configparser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "stockformer"))

from lib.Multitask_Stockformer_utils import metric, StockDataset
from lib.graph_utils import loadGraph
from Stockformermodel.Multitask_Stockformer_models import Stockformer

from src.log_parser import load_dataset as load_logs

# ── Constants ─────────────────────────────────────────────────────────────────

DATASETS     = ["nasdaq100", "wig60", "nasdaq100_extended"]
CONFIG_ROOT  = ROOT / "stockformer" / "config"
RESULTS_ROOT = ROOT / "data" / "results" / "cross_eval"

OUTFEA_CLASS   = 2
OUTFEA_REGRESS = 1


# ── Config / data helpers ─────────────────────────────────────────────────────

def _load_config(conf_path: Path) -> SimpleNamespace:
    cfg = configparser.ConfigParser()
    cfg.read(conf_path)
    return SimpleNamespace(
        cuda=cfg["train"]["cuda"],
        batch_size=int(cfg["train"]["batch_size"]),
        T1=int(cfg["data"]["T1"]),
        T2=int(cfg["data"]["T2"]),
        train_ratio=float(cfg["data"]["train_ratio"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
        test_ratio=float(cfg["data"]["test_ratio"]),
        L=int(cfg["param"]["layers"]),
        h=int(cfg["param"]["heads"]),
        d=int(cfg["param"]["dims"]),
        j=int(cfg["param"]["level"]),
        s=float(cfg["param"]["samples"]),
        w=cfg["param"]["wave"],
        traffic_file=cfg["file"]["traffic"],
        indicator_file=cfg["file"]["indicator"],
        adj_file=cfg["file"]["adj"],
        adjgat_file=cfg["file"]["adjgat"],
        model_file=cfg["file"]["model"],
        log_file=cfg["file"]["log"],
        bonus_path=cfg["file"]["bonus"],
    )


def _list_windows(dataset: str) -> list:
    """Sorted list of (label, conf_path) for every window of *dataset*."""
    conf_dir = CONFIG_ROOT / dataset
    return sorted(
        (f.stem.replace("window_", ""), f)
        for f in conf_dir.glob("window_*.conf")
    )


# ── Model helpers ─────────────────────────────────────────────────────────────

def _build_model(args: SimpleNamespace, infeature: int,
                 device: torch.device) -> Stockformer:
    return Stockformer(
        infeature, args.h * args.d,
        OUTFEA_CLASS, OUTFEA_REGRESS,
        args.L, args.h, args.d, args.s,
        args.T1, args.T2, device,
    ).to(device)


def _evaluate(model, test_ds, adjgat, batch_size: int,
              device: torch.device) -> dict:
    """Run inference on a test set; return averaged cls/reg metrics."""
    model.eval()
    n = test_ds.XL.shape[0]
    num_batch = math.ceil(n / batch_size)

    pred_cls, pred_reg = [], []
    lbl_cls,  lbl_reg  = [], []

    with torch.no_grad():
        for b in range(num_batch):
            s, e = b * batch_size, min(n, (b + 1) * batch_size)
            xl  = torch.from_numpy(test_ds.XL[s:e]).float().to(device)
            xh  = torch.from_numpy(test_ds.XH[s:e]).float().to(device)
            xc  = torch.from_numpy(test_ds.indicator_X[s:e]).float().to(device)
            te  = torch.from_numpy(test_ds.TE[s:e]).to(device)
            bon = torch.from_numpy(test_ds.bonus_X[s:e]).float().to(device)

            hc, _, hr, _ = model(xl, xh, te, bon, xc, adjgat)
            pred_cls.append(hc.cpu().numpy())
            pred_reg.append(hr.cpu().numpy())
            lbl_cls.append(test_ds.indicator_Y[s:e])
            lbl_reg.append(test_ds.Y[s:e])

    pred_cls = np.concatenate(pred_cls, axis=0)
    pred_reg = np.concatenate(pred_reg, axis=0)
    lbl_cls  = np.concatenate(lbl_cls,  axis=0)
    lbl_reg  = np.concatenate(lbl_reg,  axis=0)

    accs, maes, rmses = [], [], []
    for step in range(pred_cls.shape[1]):
        acc, mae, rmse, _ = metric(
            pred_reg[:, step, :], lbl_reg[:, step, :],
            pred_cls[:, step, :], lbl_cls[:, step, :],
        )
        accs.append(acc); maes.append(mae); rmses.append(rmse)

    return {
        "cls_acc":  float(np.mean(accs)),
        "reg_mae":  float(np.mean(maes)),
        "reg_rmse": float(np.mean(rmses)),
    }


# ── Cross-eval logic ──────────────────────────────────────────────────────────

def run_cross_eval(source_ds: str, target_ds: str,
                   device: torch.device) -> list:
    """Match windows by index, evaluate source model on target test set."""
    src_wins = _list_windows(source_ds)
    tgt_wins = _list_windows(target_ds)
    n = min(len(src_wins), len(tgt_wins))
    print(f"  Paired {n} windows  ({source_ds} → {target_ds})")

    rows = []
    for i in range(n):
        src_lbl, src_conf = src_wins[i]
        tgt_lbl, tgt_conf = tgt_wins[i]

        src_args = _load_config(src_conf)
        tgt_args = _load_config(tgt_conf)

        ckpt = Path(src_args.model_file)
        if not ckpt.exists():
            print(f"  [{i+1}/{n}] SKIP — checkpoint missing: {ckpt.name}")
            continue

        print(f"  [{i+1}/{n}] {src_lbl}  →  {tgt_lbl}", end="   ", flush=True)

        test_ds = StockDataset(tgt_args, mode="test")
        adjgat  = torch.from_numpy(loadGraph(tgt_args)).float().to(device)

        model = _build_model(src_args, test_ds.infea, device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        m = _evaluate(model, test_ds, adjgat, src_args.batch_size, device)
        print(f"cls_acc={m['cls_acc']:.4f}  "
              f"reg_mae={m['reg_mae']:.4f}  "
              f"reg_rmse={m['reg_rmse']:.4f}")

        rows.append({
            "source_dataset": source_ds,
            "target_dataset": target_ds,
            "window_index":   i,
            "source_window":  src_lbl,
            "target_window":  tgt_lbl,
            **m,
        })

    return rows


# ── CSV helper ────────────────────────────────────────────────────────────────

def _save_csv(rows: list, path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    try:
        print(f"  Saved: {path.relative_to(ROOT)}")
    except ValueError:
        print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None, choices=DATASETS)
    parser.add_argument("--target", default=None, choices=DATASETS)
    parser.add_argument("--gpu",    default=None, type=int)
    args = parser.parse_args()

    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}\n")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.source and args.target:
        pairs = [(args.source, args.target)]
    elif args.source:
        pairs = [(args.source, t) for t in DATASETS if t != args.source]
    elif args.target:
        pairs = [(s, args.target) for s in DATASETS if s != args.target]
    else:
        pairs = [(s, t) for s in DATASETS for t in DATASETS if s != t]

    # In-domain logs (from training) — used as baseline in plots
    print("Loading in-domain metrics from training logs...")
    indomain_logs = {ds: load_logs(ds) for ds in DATASETS}
    for ds, logs in indomain_logs.items():
        n = sum(1 for d in logs.values() if d.get("test"))
        print(f"  {ds}: {n} windows with test results")

    # Diagonal rows (in-domain) for heatmap
    all_rows: list[dict] = []
    for ds, logs in indomain_logs.items():
        for i, (window, d) in enumerate(sorted(logs.items())):
            if d.get("test"):
                all_rows.append({
                    "source_dataset": ds,
                    "target_dataset": ds,
                    "window_index":   i,
                    "source_window":  window,
                    "target_window":  window,
                    **d["test"],
                })

    # Cross-evaluations
    for source_ds, target_ds in pairs:
        print(f"\n{'='*60}")
        print(f"  {source_ds}  →  {target_ds}")
        print(f"{'='*60}")

        rows = run_cross_eval(source_ds, target_ds, device)
        if not rows:
            print("  No results — skipping.")
            continue

        all_rows.extend(rows)
        slug = f"{source_ds}_to_{target_ds}"
        _save_csv(rows, RESULTS_ROOT / f"{slug}.csv")

        # Console summary with delta
        print(f"\n  Summary ({source_ds} → {target_ds}):")
        for key, label in [("cls_acc", "Cls Acc"), ("reg_mae", "Reg MAE"), ("reg_rmse", "Reg RMSE")]:
            cross = [r[key] for r in rows]
            indom = [
                (indomain_logs[target_ds].get(r["target_window"], {}).get("test") or {}).get(key, float("nan"))
                for r in rows
            ]
            delta = float(np.nanmean(cross)) - float(np.nanmean(indom))
            sign  = "+" if delta >= 0 else ""
            print(f"    {label:<10} cross={np.nanmean(cross):.4f}  "
                  f"in-domain={np.nanmean(indom):.4f}  Δ={sign}{delta:.4f}")

    if all_rows:
        _save_csv(all_rows, RESULTS_ROOT / "all_results.csv")

    print(f"\nDone.  CSVs saved to:  data/results/cross_eval/")
    print("Run  python generate_cross_eval_plots.py  to generate plots.")


if __name__ == "__main__":
    main()
