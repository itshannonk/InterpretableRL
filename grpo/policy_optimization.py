import gym
from config import config_cartpole
from torch import optim
import torch.nn as nn
import numpy as np
import os
import logging
from network_utils import build_mlp, np2torch
from general import get_logger
from policy import CategoricalPolicy, GaussianPolicy

class PolicyGradient(object):
    observation_dim: int
    action_dim: int
    learning_rate: float
    seed: int
    env: gym.Env
    config: config_cartpole
    optimizer: optim.Optimizer
    policy: nn.Module
    epsilon_clip: float
    logger: logging.Logger
    avg_reward: float
    max_reward: float
    std_reward: float
    eval_reward: float

    def __init__(self, env: gym.Env, config: config_cartpole, seed: int, logger=None):
        """
        Initialize the Policy Gradient Class

        Args:
            env: gym environment
            config: config object containing hyperparameters
            seed: random seed for reproducibility
            logger: logger object for logging (default is None)
        """
        # create directory for training outputs
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        # store hyperparameters
        self.config = config
        self.seed = seed
        self.learning_rate = config.learning_rate
        self.epsilon_clip = config.eps_clip

        # set up logger
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)

        # set up environment
        self.env = env
        self.env.reset(seed=self.seed)

        # discrete vs continuous action space
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = (
            self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        )

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # initialize the policy network
        self.init_policy()

    def init_policy(self):
        """
        Initialize the policy network
        """
        mlp = build_mlp(self.observation_dim, self.action_dim, self.config.n_layers, self.config.layer_size)
        self.policy = CategoricalPolicy(mlp) if self.discrete else GaussianPolicy(mlp, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def init_averages(self):
        """
        Initialize averages to keep track of reward
        """
        self.avg_reward = 0.0
        self.max_reward = 0.0
        self.std_reward = 0.0
        self.eval_reward = 0.0

    def update_averages(self, rewards, scores_eval):
        """
        Update the averages.

        Args:
            rewards: deque
            scores_eval: list
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def train(self):
        """
        Performs training (to implement in subclass)
        """
        raise NotImplementedError

    def run(self):
        """
        Apply procedures of training for a PG.
        """
        # # record one game at the beginning
        # if self.config.record:
        #     self.record()
        # model
        self.train()
        # # record one game at the end
        # if self.config.record:
        #     self.record()




