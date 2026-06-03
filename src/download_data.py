"""
build_dataset.py
Pobiera dane z Yahoo Finance i buduje strukturę folderów
kompatybilną z Multitask-Stockformer.

Użycie:
    python data/download_data.py --dataset nasdaq100 --start 2021-01-01 --end 2023-01-01 --smoke_test

Flaga --smoke_test używa tylko 10 spółek i 1 subdataset do szybkiego testu.
"""

import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ── Listy tickerów ─────────────────────────────────────────

NASDAQ100 = [
    "NVDA","AAPL","MSFT","AMZN","GOOGL","GOOG","AVGO","TSLA","META","MU",
    "WMT","AMD","ASML","INTC","CSCO","COST","LRCX","PLTR","NFLX",
    "AMAT","TXN","QCOM","KLAC","LIN","PANW","APP","TMUS","ADI",
    "STX","PEP","CRWD","WDC","AMGN","MRVL","GILD","SHOP","HON","ISRG",
    "BKNG","PDD","VRTX","SBUX","ADBE","CEG","CDNS","FTNT","MAR","SNPS",
    "INTU","CMCSA","ADP","DDOG","MNST","MELI","CSX","NXPI","ABNB","MDLZ",
    "MPWR","ROST","ORLY","DASH","AEP","CTAS","WBD","REGN","BKR",
    "PCAR","MSTR","FANG","MCHP","FAST","EA","XEL","ADSK","ODFL",
    "EXC","IDXX","TTWO","KDP","ALNY","CCEP","PYPL","TRI","AXON","WDAY",
    "PAYX","ROP","CPRT","KHC","DXCM","GEHC","CTSH","VRSK","ZS","CHTR"
]

WIG20 = [
    "MBK.WA","BDX.WA","KTY.WA","KGH.WA","LPP.WA","CDR.WA","PEO.WA",
    "PKN.WA","PKO.WA","PGE.WA","PZU.WA","TPE.WA","KRU.WA","ALR.WA",
    "DNP.WA","ALE.WA","PCO.WA","ZAB.WA","SPL.WA","MDV.WA",
]

MWIG40 = [
    "ABE.WA","EAT.WA","ACP.WA","DOM.WA","EUR.WA","BHW.WA","ING.WA",
    "CAR.WA","DVL.WA","MIL.WA","NEU.WA","OPL.WA","RBW.WA","CPS.WA",
    "ASB.WA","AGT.WA","ENA.WA","MRB.WA","ASE.WA","MBR.WA","GPW.WA",
    "BFT.WA","JSW.WA","SNT.WA","NWG.WA","TXT.WA","WPL.WA","BNP.WA",
    "XTB.WA","APT.WA","CBF.WA","TSG.WA","VRC.WA","CRI.WA","GPP.WA",
    "DIG.WA","PXM.WA","SNK.WA","LBW.WA","PEP.WA",
]

WIG60 = WIG20 + MWIG40

DATASETS = {
    "nasdaq100":          NASDAQ100,
    "wig60":              WIG60,
    "nasdaq100_extended": NASDAQ100 + ["GLD","SLV","TLT","SHY"],
    "smoke_test":         NASDAQ100[:10],   # tylko 10 spółek
}

# ── Parametry rolling window (z oryginalnej pracy) ─────────
TRAIN_DAYS = 486
VAL_DAYS   = 81
TEST_DAYS  = 81
WINDOW     = TRAIN_DAYS + VAL_DAYS + TEST_DAYS  # 648 dni

# ── Alpha360: 6 kategorii × 60 czynników ──────────────────
ALPHA_CATEGORIES = ["CLOSE", "OPEN", "HIGH", "LOW", "VWAP", "VOLUME"]


def download_raw(tickers, start, end):
    """Pobiera surowe dane OHLCV z Yahoo Finance."""
    print(f"  Pobieram {len(tickers)} tickerów ({start} → {end})...")
    df = yf.download(
        tickers, start=start, end=end,
        interval="1d", auto_adjust=True,
        group_by="ticker", threads=True, progress=False
    )
    # Upewnij się że mamy MultiIndex nawet dla 1 tickera
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([[tickers[0]], df.columns])
    return df


def clean_tickers(df, tickers):
    """Usuwa tickery z >20% brakujących danych i forward-filluje resztę."""
    good = []
    for t in tickers:
        if t not in df.columns.get_level_values(0):
            continue
        close = df[t]["Close"]
        missing_pct = close.isna().mean()
        if missing_pct > 0.20:
            print(f"    Pomijam {t}: {missing_pct:.1%} braków")
            continue
        good.append(t)
    df = df[good].copy()
    # Forward fill potem backward fill dla brakujących
    df = df.ffill().bfill()
    return df, good


def compute_returns(close_df):
    """Dzienne stopy zwrotu: (close_t - close_{t-1}) / close_{t-1}"""
    return close_df.pct_change().fillna(0)


def compute_trend(returns_df):
    """Trend: 1 jeśli return > 0, else 0"""
    return (returns_df > 0).astype(float)


def build_alpha360(df, tickers):
    """
    Buduje 360 czynników Alpha360.
    Każdy czynnik: Ref(X, t) / close_t  dla t in 0..59
    Returns dict: {"CLOSE0": DataFrame(T x N), ...}
    """
    factors = {}
    close = df.xs("Close", axis=1, level=1)[tickers]

    for cat in ALPHA_CATEGORIES:
        if cat == "VOLUME":
            base = df.xs("Volume", axis=1, level=1)[tickers]
            for lag in range(60):
                key = f"{cat}{lag}"
                shifted = base.shift(lag)
                factors[key] = shifted / (base + 1e-12)
        elif cat == "VWAP":
            # VWAP przybliżamy jako (High+Low+Close)/3 jeśli brak osobnego
            try:
                base = df.xs("VWAP", axis=1, level=1)[tickers]
            except KeyError:
                high  = df.xs("High",  axis=1, level=1)[tickers]
                low   = df.xs("Low",   axis=1, level=1)[tickers]
                base  = (high + low + close) / 3
            for lag in range(60):
                key = f"{cat}{lag}"
                factors[key] = base.shift(lag) / close
        else:
            col_map = {"CLOSE": "Close", "OPEN": "Open",
                       "HIGH": "High",   "LOW":  "Low"}
            base = df.xs(col_map[cat], axis=1, level=1)[tickers]
            for lag in range(60):
                key = f"{cat}{lag}"
                factors[key] = base.shift(lag) / close

    # Zastąp inf i nan zerem
    for k in factors:
        factors[k] = factors[k].replace([np.inf, -np.inf], np.nan).fillna(0)

    return factors


def neutralize_factors(factors, returns):
    """
    Prosta neutralizacja: z-score per dzień (cross-sectional)
    (uproszczenie neutralizacji branżowej z pracy)
    """
    neutralized = {}
    for k, df in factors.items():
        neu = df.sub(df.mean(axis=1), axis=0)
        std = df.std(axis=1).replace(0, 1)
        neu = neu.div(std, axis=0)
        neutralized[k] = neu.clip(-3, 3)
    return neutralized


def compute_corr_matrix(returns_df):
    """Macierz korelacji Spearmana między spółkami."""
    corr, _ = spearmanr(returns_df.values)
    if returns_df.shape[1] == 1:
        corr = np.array([[1.0]])
    corr = np.nan_to_num(corr, nan=0.0)
    return corr.astype(np.float32)


def simple_graph_embedding(corr_matrix, dim=128):
    """
    Uproszczony embedding grafu przez SVD macierzy korelacji.
    Zastępuje Struc2Vec z oryginału (wymaga osobnego narzędzia).
    """
    U, s, _ = np.linalg.svd(corr_matrix)
    k = min(dim, len(s))
    embedding = U[:, :k] * np.sqrt(s[:k])
    # Normalizacja wierszy
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    embedding = (embedding / norms).astype(np.float32)
    # Zero-pad to (N, dim) when N < dim
    if k < dim:
        embedding = np.pad(embedding, ((0, 0), (0, dim - k)))
    return embedding


def generate_config(config_path, window_dir, label, out_root):
    """Generates a Stockformer .conf for one rolling window."""
    window_dir = os.path.abspath(window_dir).replace("\\", "/")
    alpha_dir  = f"{window_dir}/Alpha_360_{label}"
    cpt_dir    = os.path.abspath(os.path.join(out_root, "cpt")).replace("\\", "/")
    log_dir    = os.path.abspath(os.path.join(out_root, "log")).replace("\\", "/")
    os.makedirs(cpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_ratio = round(TRAIN_DAYS / WINDOW, 6)
    val_ratio   = round(VAL_DAYS   / WINDOW, 6)
    test_ratio  = round(TEST_DAYS  / WINDOW, 6)

    content = f"""[file]
traffic = {window_dir}/flow.npz
indicator = {window_dir}/trend_indicator.npz
adj = {window_dir}/corr_adj.npy
adjgat = {window_dir}/128_corr_struc2vec_adjgat.npy
model = {cpt_dir}/saved_model_{label}
log = {log_dir}/log_{label}
bonus = {alpha_dir}

[data]
dataset = STOCK
T1 = 20
T2 = 2
train_ratio = {train_ratio}
val_ratio = {val_ratio}
test_ratio = {test_ratio}

[train]
cuda = 0
max_epoch = 100
batch_size = 12
learning_rate = 0.001
seed = 1

[param]
layers = 2
heads = 1
dims = 128
samples = 1
wave = sym2
level = 1
"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write(content)


def save_subdataset(out_dir, dates, returns, trend, factors, corr, emb):
    """Zapisuje jeden subdataset w formacie Stockformera."""
    os.makedirs(out_dir, exist_ok=True)

    # Wycinamy dane dla tego okresu
    ret_slice   = returns.loc[dates]
    trend_slice = trend.loc[dates]

    # flow.npz: shape (T, N)
    flow_data = ret_slice.values.astype(np.float32)
    np.savez(os.path.join(out_dir, "flow.npz"), data=flow_data)

    # trend_indicator.npz: shape (T, N)
    trend_data = trend_slice.values.astype(np.float32)
    np.savez(os.path.join(out_dir, "trend_indicator.npz"), data=trend_data)

    # label_processed.csv: (T x N) stopy zwrotu
    ret_slice.to_csv(os.path.join(out_dir, "label_processed.csv"))

    # corr_adj.npy: (N, N)
    np.save(os.path.join(out_dir, "corr_adj.npy"), corr)

    # 128_corr_struc2vec_adjgat.npy: (N, 128)
    np.save(os.path.join(out_dir, "128_corr_struc2vec_adjgat.npy"), emb)

    # Alpha_360 folder
    alpha_dir = os.path.join(out_dir, f"Alpha_360_{dates[0].strftime('%Y-%m-%d')}_{dates[-1].strftime('%Y-%m-%d')}")
    os.makedirs(alpha_dir, exist_ok=True)

    for factor_name, factor_df in factors.items():
        factor_slice = factor_df.loc[dates]
        factor_slice.to_csv(os.path.join(alpha_dir, f"{factor_name}.csv"))


def build_rolling_windows(dates, n_windows=14):
    """Generuje rolling windows zgodnie z metodologią z pracy."""
    windows = []
    total = len(dates)
    
    if total < WINDOW:
        print(f"  BŁĄD: Za mało dni ({total}) dla okna ({WINDOW})")
        return windows

    if n_windows == 1:
        steps = [0]
    else:
        step = (total - WINDOW) // (n_windows - 1)
        steps = [i * step for i in range(n_windows)]

    for start_idx in steps:
        end_idx = start_idx + WINDOW
        if end_idx > total:
            break
        w_dates = dates[start_idx:end_idx]
        windows.append({
            "train": w_dates[:TRAIN_DAYS],
            "val":   w_dates[TRAIN_DAYS:TRAIN_DAYS + VAL_DAYS],
            "test":  w_dates[TRAIN_DAYS + VAL_DAYS:],
            "label": f"{w_dates[0].strftime('%Y-%m-%d')}_{w_dates[-1].strftime('%Y-%m-%d')}"
        })
    return windows


def main(args):
    dataset_name = "smoke_test" if args.smoke_test else args.dataset
    tickers = DATASETS[dataset_name]
    out_root = os.path.join(args.out_dir, dataset_name)
    os.makedirs(out_root, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Dataset:  {dataset_name} ({len(tickers)} tickerów)")
    print(f"Okres:    {args.start} → {args.end}")
    print(f"Output:   {out_root}")
    print(f"{'='*60}\n")

    # 1. Pobierz dane
    raw = download_raw(tickers, args.start, args.end)
    raw, tickers = clean_tickers(raw, tickers)
    print(f"  Zostało {len(tickers)} tickerów po czyszczeniu\n")

    # 2. Oblicz returns i trend
    close = raw.xs("Close", axis=1, level=1)[tickers]
    returns = compute_returns(close).iloc[1:]   # usuń pierwszy NaN wiersz
    trend   = compute_trend(returns)
    dates   = returns.index
    print(f"  Liczba dni handlowych: {len(dates)}")
    print(f"  Potrzeba minimum: {WINDOW} dni ({TRAIN_DAYS}+{VAL_DAYS}+{TEST_DAYS})")

    # 3. Oblicz Alpha360
    print("\n  Buduję czynniki Alpha360...")
    factors_raw = build_alpha360(raw, tickers)
    factors     = neutralize_factors(factors_raw, returns)

    # Przytnij też factors do tych samych dat co returns
    factors = {k: v.loc[dates] for k, v in factors.items()}

    # 4. Macierz korelacji i embedding (na całym zbiorze)
    print("  Obliczam macierz korelacji...")
    corr = compute_corr_matrix(returns)
    emb  = simple_graph_embedding(corr, dim=128)

    # 5. Rolling windows
    n_windows = 1 if args.smoke_test else 14
    windows = build_rolling_windows(dates, n_windows)
    print(f"\n  Generuję {len(windows)} subdataset(ów)...\n")

    for i, w in enumerate(tqdm(windows, desc="Subdatasety")):
        # Połącz wszystkie daty okna
        all_dates = w["train"].append(w["val"]).append(w["test"])
        folder_name = f"Stock_{dataset_name.upper()}_{w['label']}"
        out_dir = os.path.join(out_root, folder_name)

        # Przelicz korelację tylko na danych treningowych
        corr_w = compute_corr_matrix(returns.loc[w["train"]])
        emb_w  = simple_graph_embedding(corr_w, dim=128)

        save_subdataset(
            out_dir   = out_dir,
            dates     = all_dates,
            returns   = returns,
            trend     = trend,
            factors   = factors,
            corr      = corr_w,
            emb       = emb_w,
        )

        config_path = os.path.join(args.config_dir, dataset_name, f"window_{w['label']}.conf")
        generate_config(config_path, out_dir, w["label"], out_root)

    print(f"\n✓ Gotowe! Dane zapisane w: {out_root}")
    print(f"  Spółki: {len(tickers)}")
    print(f"  Subdatasety: {len(windows)}")
    if windows:
        print(f"  Przykładowy folder: {os.path.join(out_root, windows[0]['label'])}")
    else:
        print("  BŁĄD: Nie wygenerowano żadnych okien!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="smoke_test",
                        choices=list(DATASETS.keys()))
    parser.add_argument("--start",      default="2021-01-01")
    parser.add_argument("--end",        default="2026-01-31")
    parser.add_argument("--out_dir",    default="data/processed")
    parser.add_argument("--config_dir", default="stockformer/config")
    parser.add_argument("--smoke_test", action="store_true",
                        help="10 spółek, 1 subdataset – szybki test")
    args = parser.parse_args()

    if args.dataset == "smoke_test":
        args.smoke_test = True

    main(args)