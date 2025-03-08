# from ppo import PPO
# from grpo import GRPO
# import argparse
import numpy as np
import torch
import gym
# import pybullet_envs
from config import config_cartpole, config_mountaincar
import random
from policy_optimization import PolicyGradient
# from ppo import PPO
from grpo import GRPO

# import logging
# logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Set up the policy
seed = 1
torch.random.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# get config
config = config_mountaincar(ppo=False, seed=seed)
env = gym.make(config.env_name)

# train model
# model = PolicyGradient(env, config, seed)
model = GRPO(env, config, seed)
model.run()
print("created model")

env.close()