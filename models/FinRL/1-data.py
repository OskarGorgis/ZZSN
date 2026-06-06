"""
Stock NeurIPS2018 Part 1. Data

This series is a reproduction of paper "Deep reinforcement learning for automated stock trading: An ensemble strategy".

Introduce how to use FinRL to fetch and process data that we need for ML/RL trading.
"""

from __future__ import annotations

import itertools
import argparse

import pandas as pd

from configs import config_tickers
from configs.config import INDICATORS
from configs.config import TRADE_END_DATE
from configs.config import TRADE_START_DATE
from configs.config import TRAIN_END_DATE
from configs.config import TRAIN_START_DATE
from configs.config import TRAIN_START_DATE
from configs.config import DATA_SAVE_DIR
from finrl.meta.preprocessor.preprocessors import data_split
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

def prepare_stock_data(ticker=config_tickers.NAS_100_TICKER, train_save_path="data/NASDAQ100_train_data.csv"):
    print("\n=== Downloading Tickers ===")
    print(ticker)

    df_raw = YahooDownloader(
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=ticker,
    ).fetch_data()
    print("\n=== Raw data ===")
    print(df_raw.head())


    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=True,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df_raw)

    list_ticker = processed["tic"].unique().tolist()
    list_date = list(
        pd.date_range(processed["date"].min(), processed["date"].max()).astype(str)
    )
    combination = list(itertools.product(list_date, list_ticker))

    processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(
        processed, on=["date", "tic"], how="left"
    )
    processed_full = processed_full[processed_full["date"].isin(processed["date"])]
    processed_full = processed_full.sort_values(["date", "tic"])
    processed_full = processed_full.fillna(0)

    print("\n=== Processed data ===")
    print(processed_full.head())


    train = data_split(processed_full, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(processed_full, TRADE_START_DATE, TRADE_END_DATE)
    print(f"\nTrain data length: {len(train)}")
    print(f"Trade data length: {len(trade)}")

    train.to_csv(train_save_path)
    trade.to_csv(trade_save_path := train_save_path.replace("train", "trade"))
    print(f"Data saved to {train_save_path} and {trade_save_path}")

def main(DATA_DIR="data"):
    # config mapping
    dataset_configs = {
        "nasdaq100": (config_tickers.NAS_100_TICKER, f"{DATA_SAVE_DIR}/NASDAQ100_train_data.csv"),
        "nasdaq100_extended": (config_tickers.NAS_100_EXT_TICKER, f"{DATA_SAVE_DIR}/NASDAQ100_EXT_train_data.csv"),
        "wig60": (config_tickers.WIG60_TICKER, f"{DATA_SAVE_DIR}/WIG60_train_data.csv"),
        "csi300": (config_tickers.CSI300_TICKER, f"{DATA_SAVE_DIR}/CSI300_train_data.csv")
    }

    parser = argparse.ArgumentParser(description="Fetch and preprocess stock data for FinRL.")
    parser.add_index = False
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=list(dataset_configs.keys()) + ["all"], 
        default="all",
        help="The dataset to download and process (default: all)"
    )
    args = parser.parse_args()


    if args.dataset == "all":
        for ticker, save_path in dataset_configs.values():
            prepare_stock_data(ticker, train_save_path=save_path)
    else:
        ticker, save_path = dataset_configs[args.dataset]
        prepare_stock_data(ticker, train_save_path=save_path)
if __name__ == "__main__":
    main()