#We will define a custom OpenAI Gym environment for our trading strategy.

import gymnasium as gym
class TradingEnv(gym.Env):
    """Custom Environment for trading that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df
        self.max_steps = len(df) - 1
        self.current_step = 0

        # Define action and observation space
        # Actions: Sell, Hold, Buy
        self.action_space = spaces.Discrete(3)

        # Observations: Spread, Spread Volatility, Correlation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )

        # Initial balance and positions
        self.balance = 100000
        self.positions = 0  # Number of pairs traded
        self.position_value = 0
        self.trades = []


    def _next_observation(self):
        obs = np.array([
            self.df.iloc[self.current_step, self.df.columns.get_loc('Spread')],
            self.df.iloc[self.current_step, self.df.columns.get_loc('Spread_vol')],
            self.df.iloc[self.current_step, self.df.columns.get_loc('Correlation')]
        ])
        return obs

    def step(self, action):
        done = False
        reward = 0

        # Get the current price spread
        current_spread = self.df.loc[self.current_step, 'Spread']

        # Define transaction cost
        transaction_cost = 0.001  # 0.1% per trade

        if action == 0:  # Sell the spread
            self.positions -= 1
            self.balance += current_spread * (1 - transaction_cost)
            self.trades.append(-1)
        elif action == 2:  # Buy the spread
            self.positions += 1
            self.balance -= current_spread * (1 + transaction_cost)
            self.trades.append(1)
        else:  # Hold
            self.trades.append(0)

        # Calculate reward (Profit and Loss)
        reward = self.positions * (self.df.loc[self.current_step + 1, 'Spread'] - current_spread)

        # Penalize excessive trading
        reward -= transaction_cost * abs(action - 1) * current_spread

        # Move to the next step
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # Get next observation
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        self.balance = 100000
        self.positions = 0
        self.current_step = 0
        self.trades = []
        return self._next_observation()

    def render(self, mode='human', close=False):
        profit = self.balance + self.positions * self.df.loc[self.current_step, 'Spread'] - 100000
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Positions: {self.positions}')
        print(f'Profit: {profit}')
