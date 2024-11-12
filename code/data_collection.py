# This file gets data for stock pairs and generates synthetic data for training
# Import libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import optuna
import torch as th
import random


# Define the tickers
tickers = ['RSP', 'SPY']  # Change tickers according to need

# Download historical data
data = yf.download(tickers, start='2010-01-01', end='2023-01-01')['Adj Close']

# Display the head of the data
data.head()


def generate_gbm(start_price, mu, sigma, days):
    dt = 1/252
    prices = []
    price = start_price
    for _ in range(days):
        price *= np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
        prices.append(price)
    return prices

# Generate synthetic data for two assets
np.random.seed(42)
synthetic_length = len(data)
synthetic_data = pd.DataFrame({
    'Asset1': generate_gbm(100, 0.05, 0.2, synthetic_length),
    'Asset2': generate_gbm(100, 0.03, 0.25, synthetic_length)
})

# Combine real and synthetic data
synthetic_data = synthetic_data.reset_index(drop=True)
data = data.reset_index(drop=True)
data = pd.concat([data, synthetic_data], axis=1)

# Display the head of the combined data
print("\nCombined Data:")
print(data.head())
