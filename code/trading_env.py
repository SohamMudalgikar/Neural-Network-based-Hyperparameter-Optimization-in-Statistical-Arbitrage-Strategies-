#We will define a custom OpenAI Gym environment for our trading strategy.

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnv(gym.Env):
    """Custom Environment for trading that follows gymnasium interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()

        if df.empty:
            raise ValueError("The DataFrame 'df' is empty. Please check the data preprocessing steps.")

        self.df = df.reset_index(drop=True)
        self.max_steps = len(self.df) - 1
        self.current_step = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        # Initial balance and positions
        self.balance = 100000
        self.positions = 0
        self.trades = []

    def _next_observation(self):
        obs = np.array([
            self.df.loc[self.current_step, 'Spread'],
            self.df.loc[self.current_step, 'Spread_vol'],
            self.df.loc[self.current_step, 'Correlation']
        ], dtype=np.float32)
        return obs

    def step(self, action):
        done = False

        current_spread = self.df.loc[self.current_step, 'Spread']
        transaction_cost = 0.001

        if action == 0:
            self.positions -= 1
            self.balance += current_spread * (1 - transaction_cost)
            self.trades.append(-1)
        elif action == 2:
            self.positions += 1
            self.balance -= current_spread * (1 + transaction_cost)
            self.trades.append(1)
        else:
            self.trades.append(0)

        if self.current_step < self.max_steps:
            next_spread = self.df.loc[self.current_step + 1, 'Spread']
            pnl = self.positions * (next_spread - current_spread)
        else:
            pnl = 0

        # Reward function with risk adjustment
        reward = pnl - transaction_cost * abs(action - 1) * current_spread

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        obs = self._next_observation()

        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = 100000
        self.positions = 0
        self.current_step = 0
        self.trades = []
        return self._next_observation(), {}

    def render(self, mode='human'):
        profit = self.balance + self.positions * self.df.loc[self.current_step, 'Spread'] - 100000
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Positions: {self.positions}')
        print(f'Profit: {profit}')
