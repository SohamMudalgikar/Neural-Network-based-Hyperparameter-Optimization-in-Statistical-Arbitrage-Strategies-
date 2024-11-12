# We will use Optuna for Bayesian optimization to find the best hyperparameters for the PPO agent.

import optuna
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from trading_env import TradingEnv
import pandas as pd

def optimize_agent(trial):
    n_steps = trial.suggest_int('n_steps', 1024, 2048, step=256)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)

    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    data = pd.read_csv('data/processed_data.csv')
    env = DummyVecEnv([lambda: TradingEnv(data)])

    # Adjust batch_size to be a factor of n_steps
    batch_size = 256

    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        policy_kwargs=policy_kwargs
    )

    model.learn(total_timesteps=20000)

    # Evaluate the agent
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        total_reward += rewards[0]
        done = dones[0]

    return float(total_reward)

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=20)
    print('Best hyperparameters: ', study.best_params)
    # Save the best hyperparameters
    with open('results/best_hyperparameters.txt', 'w') as f:
        f.write(str(study.best_params))
