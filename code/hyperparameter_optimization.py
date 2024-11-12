# We will use Optuna for Bayesian optimization to find the best hyperparameters for the PPO agent.
def optimize_agent(trial):
    """Optimize PPO hyperparameters using Optuna"""

    # Hyperparameters to tune
    n_steps = trial.suggest_int('n_steps', 128, 2048)
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1, log=True)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 0.1, log=True)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.4)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)

    # policy_kwargs = dict(
    #     activation_fn=th.nn.Tanh,
    #     net_arch=[dict(pi=[64, 64], vf=[64, 64])]
    # )

    # Modify policy_kwargs
    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=[dict(pi=[128, 128, 64], vf=[128, 128, 64])]
    )

    # Create environment
    env = DummyVecEnv([lambda: TradingEnv(data)])

    # Initialize the agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log="./tensorboard/",
        n_steps=n_steps,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        policy_kwargs=policy_kwargs
    )

    # Train the agent
    model.learn(total_timesteps=10000)

    # Evaluate the agent
    obs = env.reset()
    total_reward = 0
    for i in range(len(data) - 1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        total_reward += rewards
        if done:
            break

    return total_reward

# Run the Optimization
study = optuna.create_study(direction='maximize')
study.optimize(optimize_agent, n_trials=10)

print('Best hyperparameters: ', study.best_params)
