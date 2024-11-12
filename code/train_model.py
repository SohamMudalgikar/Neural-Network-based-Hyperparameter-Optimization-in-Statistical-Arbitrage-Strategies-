# Train the PPO Agent with Optimized Hyperparameters
# Using the best hyperparameters from Optuna, we will train the PPO agent.
# Get the best hyperparameters

import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
# from trading_env import TradingEnv

if __name__ == "__main__":
    # Load data and best hyperparameters
    data = pd.read_csv('data/processed_data.csv')
    with open('results/best_hyperparameters.txt', 'r') as f:
        best_params = eval(f.read())

    env = DummyVecEnv([lambda: TradingEnv(data)])

    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=best_params['n_steps'],
        batch_size=256,
        gamma=best_params['gamma'],
        learning_rate=best_params['learning_rate'],
        ent_coef=best_params['ent_coef'],
        clip_range=best_params['clip_range'],
        gae_lambda=best_params['gae_lambda'],
        policy_kwargs=policy_kwargs
    )

    model.learn(total_timesteps=100000)
    model.save('results/model.zip')

