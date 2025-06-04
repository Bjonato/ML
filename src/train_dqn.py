"""Example training script for DQN on the TradingEnv."""

import os, sys
sys.path.append(os.path.dirname(__file__))

import argparse
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from trading_env import TradingEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="CSV file with price and volume")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--save-path", type=str, default="dqn_trading_model")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.data)

    env = DummyVecEnv([lambda: TradingEnv(df)])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        verbose=1,
    )

    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)


if __name__ == "__main__":
    main()
