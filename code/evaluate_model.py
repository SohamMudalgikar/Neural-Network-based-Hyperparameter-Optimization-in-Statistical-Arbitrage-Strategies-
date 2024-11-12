# We will test the trained model on the environment and visualize the performance.


import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# from trading_env import TradingEnv

if __name__ == "__main__":
    data = pd.read_csv('data/processed_data.csv')
    env = DummyVecEnv([lambda: TradingEnv(data)])
    model = PPO.load('results/model.zip', env=env)

    obs = env.reset()
    env.envs[0].trades = []

    profits = []
    positions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        current_profit = env.envs[0].balance + env.envs[0].positions * data.loc[env.envs[0].current_step, 'Spread'] - 100000
        profits.append(current_profit)
        positions.append(env.envs[0].positions)
        done = dones[0]

    # Plot the results
    plt.figure(figsize=(12,6))
    plt.subplot(2,1,1)
    plt.plot(profits)
    plt.title('Cumulative Profit')
    plt.xlabel('Time Steps')
    plt.ylabel('Profit')

    plt.subplot(2,1,2)
    plt.plot(positions)
    plt.title('Positions Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Positions')

    plt.tight_layout()
    plt.savefig('results/figures/performance_plots.png')
    plt.show()

    # Additional Visualizations
    # Price Series
    plt.figure(figsize=(12,6))
    plt.plot(data['JPM'], label='JPM')
    plt.plot(data['BAC'], label='BAC')
    plt.title('Price Series')
    plt.xlabel('Time Steps')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.savefig('results/figures/price_series.png')
    plt.show()

    # Spread
    plt.figure(figsize=(12,6))
    plt.plot(data['Spread'], label='Spread')
    plt.title('Price Spread Between JPM and BAC')
    plt.xlabel('Time Steps')
    plt.ylabel('Spread')
    plt.legend()
    plt.savefig('results/figures/spread_plot.png')
    plt.show()

    # Spread Volatility
    plt.figure(figsize=(12,6))
    plt.plot(data['Spread_vol'], label='Spread Volatility')
    plt.title('Spread Volatility')
    plt.xlabel('Time Steps')
    plt.ylabel('Volatility')
    plt.legend()
    plt.savefig('results/figures/spread_volatility.png')
    plt.show()

    # Correlation
    plt.figure(figsize=(12,6))
    plt.plot(data['Correlation'], label='Correlation')
    plt.title('Rolling Correlation Between JPM and BAC')
    plt.xlabel('Time Steps')
    plt.ylabel('Correlation')
    plt.legend()
    plt.savefig('results/figures/correlation.png')
    plt.show()
