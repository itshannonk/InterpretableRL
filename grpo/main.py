# from ppo import PPO
# from grpo import GRPO
# import argparse
import numpy as np
import torch
import gym
# import pybullet_envs
from config import config_cartpole
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
config = config_cartpole(ppo=False, seed=seed)
env = gym.make(config.env_name)

# train model
# model = PolicyGradient(env, config, seed)
model = GRPO(env, config, seed)
model.run()
print("created model")





# env = gym.make("CartPole-v1")

# Evaluation
num_episodes = 1000

for episode in range(num_episodes):
    # Reset the environment
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # randomly sample an action -- sample this from the policy
        action = env.action_space.sample()

        # take the action
        state, reward, done, truncated, info = env.step(action)
        total_reward += reward


    print(f"Episode {episode + 1}: Total Reward: {total_reward}")

env.close()