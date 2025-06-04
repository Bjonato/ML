import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """Custom Trading Environment for reinforcement learning.

    The environment expects a pandas DataFrame with the columns
    ``open``, ``high``, ``low``, ``close`` and ``volume``. Values are
    normalized by the first row so the agent sees inputs roughly in the
    ``0-1`` range.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000.0):
        super().__init__()
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        # Observation: open, high, low, close, volume (normalized)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.reset()

    def _get_observation(self) -> np.ndarray:
        row = self.data.loc[self.current_step]

        # Simple normalization by dividing by initial value of each feature
        open_norm = row["open"] / self.data.loc[0, "open"]
        high_norm = row["high"] / self.data.loc[0, "high"]
        low_norm = row["low"] / self.data.loc[0, "low"]
        close_norm = row["close"] / self.data.loc[0, "close"]
        volume_norm = row["volume"] / (self.data.loc[0, "volume"] + 1e-8)

        return np.array(
            [open_norm, high_norm, low_norm, close_norm, volume_norm],
            dtype=np.float32,
        )

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0  # -1=short, 0=flat, 1=long
        self.entry_price = 0.0
        self.current_step = 0
        return self._get_observation()

    def step(self, action: int):
        done = False
        reward = 0.0
        price = self.data.loc[self.current_step, "close"]

        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = price
        elif action == 2:  # Sell
            if self.position == 1:
                reward = price - self.entry_price
                self.balance += reward
                self.position = 0
                self.entry_price = 0.0

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        obs = self._get_observation()
        info = {"balance": self.balance, "position": self.position}
        return obs, reward, done, info

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}")
