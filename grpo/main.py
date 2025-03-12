# from ppo import PPO
# from grpo import GRPO
# import argparse
import numpy as np
import torch
# import pybullet_envs
from config import setup_env
import random
from policy_optimization import PolicyGradient
# from ppo import PPO
from grpo import GRPO
from ppo import PPO

# import logging
# logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

env_name = 'Humanoid-v3'
seeds = range(1, 2)

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





# # Set up the policy
# seed = 6
# torch.random.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

# config, env = setup_env("CartPole-v1", grpo=False, seed=seed, trace_memory=True)
# # # get config
# # config = config_cartpole(grpo=True, seed=seed)
# # env = gym.make(config.env_name)

# # train model
# # model = PolicyGradient(env, config, seed)
# model = PPO(env, config, seed)
# model.run()
# print("created model")

# env.close()