"""
Train and run the GRPO and PPO algorithms on your desired environment (defined in config.py).
"""
import numpy as np
import torch
from config import setup_env
import random
from policy_optimization import PolicyGradient
from grpo import GRPO
from ppo import PPO

# Replace 'Reacher-v5' with your desired environment
env_name = 'Reacher-v5'
seeds = range(1, 4)

for seed in seeds:
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Running GRPO on {env_name} with seed {seed}")
    config, env = setup_env(env_name, grpo=True, seed=seed, trace_memory=True)
    model = GRPO(env, config, seed)
    model.run()
    env.close()

    print(f"Running PPO on {env_name} with seed {seed}")
    config, env = setup_env(env_name, grpo=False, seed=seed, trace_memory=True)
    model = PPO(env, config, seed)
    model.run()
    env.close()
