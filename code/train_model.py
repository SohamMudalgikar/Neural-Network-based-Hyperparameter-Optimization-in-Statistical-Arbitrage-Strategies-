# Train the PPO Agent with Optimized Hyperparameters
# Using the best hyperparameters from Optuna, we will train the PPO agent.
# Get the best hyperparameters
best_params = study.best_params

# Create environment
env = DummyVecEnv([lambda: TradingEnv(data)])

# Initialize the agent with best hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard/",
    n_steps=best_params['n_steps'],
    gamma=best_params['gamma'],
    learning_rate=best_params['learning_rate'],
    ent_coef=best_params['ent_coef'],
    clip_range=best_params['clip_range'],
    gae_lambda=best_params['gae_lambda'],
    policy_kwargs=dict(
        activation_fn=th.nn.Tanh,
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]
    )
)

# Train the agent
model.learn(total_timesteps=50000)
