"""
Stock NeurIPS2018 Part 2. Train

This series is a reproduction of paper "Deep reinforcement learning for
automated stock trading: An ensemble strategy".

Introduce how to use FinRL to make data into the gym form environment, and train DRL agents on it.
"""

from __future__ import annotations
from pdb import run

import pandas as pd
from stable_baselines3.common.logger import configure

from finrl.agents.stablebaselines3.models import DRLAgent
from configs.config import INDICATORS
from configs.config import RESULTS_DIR
from configs.config import TRAINED_MODEL_DIR
from configs.config import DATA_SAVE_DIR
from finrl.main import check_and_make_directories
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

check_and_make_directories([TRAINED_MODEL_DIR])


def build_env(train_data_path=f"{DATA_SAVE_DIR}/NASDAQ100_train_data.csv"):

    train = pd.read_csv(train_data_path)
    train = train.set_index(train.columns[0])
    train.index.names = [""]

    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
    }

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(type(env_train))
    return env_train

# --- SAC ---
def train_sac_agent(env_train, model_name="agent_sac"):
    agent = DRLAgent(env=env_train)
    SAC_PARAMS = {
        "batch_size": 128,
        "buffer_size": 100000,
        "learning_rate": 0.0001,
        "learning_starts": 100,
        "ent_coef": "auto_0.1",
    }
    model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
    tmp_path = RESULTS_DIR + "/sac"
    new_logger_sac = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_sac.set_logger(new_logger_sac)

    # set total_timesteps to a small number for testing, and increase it for real training (250_000 is ok)
    trained_sac = agent.train_model(model=model_sac, tb_log_name="sac", total_timesteps=2500)

    trained_sac.save(TRAINED_MODEL_DIR + "/" + model_name)
    print(f"SAC agent trained and saved to {TRAINED_MODEL_DIR}")

def run_training():
    nasdaq100_env_train = build_env(train_data_path=f"{DATA_SAVE_DIR}/NASDAQ100_train_data.csv")
    train_sac_agent(nasdaq100_env_train, model_name="agent_sac_nasdaq100")

    nasdaq100_env_train = build_env(train_data_path=f"{DATA_SAVE_DIR}/NASDAQ100_EXT_train_data.csv")
    train_sac_agent(nasdaq100_env_train, model_name="agent_sac_nasdaq100_ext")


    wig60_env_train = build_env(train_data_path=f"{DATA_SAVE_DIR}/WIG60_train_data.csv")
    train_sac_agent(wig60_env_train, model_name="agent_sac_wig60")



if __name__ == "__main__":
    run_training()

