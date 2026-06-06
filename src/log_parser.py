"""Parse Multitask-Stockformer training logs into structured data."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_ROOT = ROOT / "data" / "stockformer" / "processed"

_AVG_PAT = re.compile(
    r"average, acc: ([^\s,]+), mae: ([^\s,]+), rmse: ([^\s,]+)"
)
_EPOCH_PAT = re.compile(r"epoch (\d+).*loss ([\d.]+)")


def parse_log(log_path: Path) -> dict:
    """Parse one training log file.

    Returns
    -------
    dict with keys:
        epochs        : list[int]
        train_loss    : list[float]
        val_cls_acc   : list[float]  - classification accuracy per epoch (val set)
        val_reg_mae   : list[float]  - regression MAE per epoch (val set)
        val_reg_rmse  : list[float]
        test          : dict | None  - {cls_acc, reg_mae, reg_rmse} from test phase
    """
    epochs, train_loss = [], []
    val_cls_acc, val_reg_mae, val_reg_rmse = [], [], []
    test_metrics = None
    in_test = False

    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if "testing begin" in line:
                in_test = True
                continue

            m = _EPOCH_PAT.match(line)
            if m:
                epochs.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                in_test = False
                continue

            m = _AVG_PAT.match(line)
            if m:
                acc  = float(m.group(1))
                mae  = float(m.group(2))
                rmse = float(m.group(3))
                if in_test:
                    test_metrics = {"cls_acc": acc, "reg_mae": mae, "reg_rmse": rmse}
                else:
                    val_cls_acc.append(acc)
                    val_reg_mae.append(mae)
                    val_reg_rmse.append(rmse)

    return {
        "epochs":       epochs,
        "train_loss":   train_loss,
        "val_cls_acc":  val_cls_acc,
        "val_reg_mae":  val_reg_mae,
        "val_reg_rmse": val_reg_rmse,
        "test":         test_metrics,
    }


def load_dataset(dataset: str) -> dict:
    """Return {window_label: parsed_log} for every window of *dataset*."""
    log_dir = DATA_ROOT / dataset / "log"
    if not log_dir.exists():
        return {}
    return {
        f.name.replace("log_", ""): parse_log(f)
        for f in sorted(log_dir.iterdir())
        if f.is_file()
    }


def load_all_datasets(datasets: list) -> dict:
    """Return {dataset: {window: parsed_log}} for all datasets."""
    return {ds: load_dataset(ds) for ds in datasets}
