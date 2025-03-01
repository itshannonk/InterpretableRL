from policy_optimization import PolicyGradient
from config import config_cartpole
import gym
import numpy as np
from network_utils import np2torch
import torch
from general import export_plot
import copy

def duplicate_env(env: gym.Env, num_envs: int, seed: int) -> list:
    """
    Duplicate the environment num_envs times.
    """
    envs = []
    for _ in range(num_envs):
        # Create en environment with the same specifications as the original env
        new_env = gym.make(env.spec.id)
        new_env.reset(seed=seed)
        # Set the state to the same as the current env
        new_env.env.state = copy.deepcopy(env.env.state)
        envs.append(new_env)

    return envs

class GRPO(PolicyGradient):
    group_size: int

    def __init__(self, env: gym.Env, config: config_cartpole, seed: int, logger=None):
        super(GRPO, self).__init__(env, config, seed, logger)
        self.group_size = config.group_size
        self.group_ep_len = config.group_episode_length

    def sample_path(self, env: gym.Env, num_episodes=None) -> tuple:
        """
        Sample paths (trajectories) from the environment.

        Return the path and
        """
        pass

    def get_returns(self, paths):
        """
        Compute the discounted future returns for each time step in each path
        """
        pass

    def normalize_group_rewards(self, rewards: np.array) -> np.array:
        """
        Normalize the advantages to have mean 0 and std 1 across each group

        Args:
            rewards: np.array of shape [batch size, ep_length, group size]
        
        Returns:
            np.array of shape [batch size, ep_length, group size]
        """
        mean, std = np.mean(rewards, axis=2), np.std(rewards, axis=2)
        return (rewards - mean[:, :, None]) / std[:, :, None]

    def calculate_advantage(self, observations: np.array, group_observations: np.array) -> np.array:
        """
        Calculate the advantages for each time step in each path
        """
        pass

    def update_policy(self, observations: np.array, actions: np.array, advantages: np.array, old_logprobs: np.array):
        """
        Do a gradient update using the GRPO loss function

        Args:
            observations: np.array of shape [batch size, group size, state dim]
            actions: np.array of shape [batch size, group size]
            advantages: np.array of shape [batch size, group size]
            old_logprobs: np.array of shape [batch size, group size]

        Perform one update on the policy using the provided data using the PPO clipped
        objective function.
        """
        # loop through groups to compute the loss
        loss = 0.0
        for i in range(self.config.batch_size):
            observations = np2torch(observations[i])
            actions = np2torch(actions[i])
            advantages = np2torch(advantages[i])
            old_logprobs = np2torch(old_logprobs[i])

            # Here, I'm treating a group as a batch of simple observations. Is that right?
            new_logprobs = self.policy.action_distribution(observations).log_prob(actions)
            prob_ratio = torch.exp(new_logprobs - old_logprobs)
            clip = torch.clamp(prob_ratio, 1 - self.eps_clip, 1 + self.eps_clip)
            loss += -torch.min(prob_ratio * advantages, clip * advantages).mean()

        loss = loss / self.group_size

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        """
        Performs training

        You do not have to change or use anything here, but take a look
        to see how all the code you've written fits together!
        """
        last_record = 0

        self.init_averages()
        all_total_rewards = (
            []
        )  # the returns of all episodes samples for training purposes
        averaged_total_rewards = []  # the returns for each iteration

        for t in range(self.config.num_batches):

            # collect a minibatch of samples
            paths, total_rewards, group_paths = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            old_logprobs = np.concatenate([path["old_logprobs"] for path in paths])

            group_observations = np.concatenate([path["observations"] for path in group_paths])
            group_actions = np.concatenate([path["actions"] for path in group_paths])
            group_rewards = np.concatenate([path["rewards"] for path in group_paths])
            group_old_logprobs = np.concatenate([path["old_logprobs"] for path in group_paths])

            # compute Q-val estimates (discounted future returns) for each time step
            # ISSUE: we can't calculate returns since we don't step each group all the way to the end,
            # so we can't calculate the returns for each step in the path
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            group_rewards = self.normalize_group_rewards(group_rewards)

            # run training operations
            for k in range(self.config.update_freq):
                # self.baseline_network.update_baseline(returns, observations)
                self.update_policy(observations, actions, advantages, 
                                   old_logprobs)

            # logging
            if t % self.config.summary_freq == 0:
                self.update_averages(total_rewards, all_total_rewards)
                # self.record_summary(t)

            # compute reward statistics for this batch and log
            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "[ITERATION {}]: Average reward: {:04.2f} +/- {:04.2f}".format(
                    t, avg_reward, sigma_reward
            )
            averaged_total_rewards.append(avg_reward)
            self.logger.info(msg)

            if self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                # self.record()

        self.logger.info("- Training done.")
        np.save(self.config.scores_output, averaged_total_rewards)
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config.env_name,
            self.config.plot_output,
        )

    def sample_group(self, env: gym.Env, current_state, num_episodes=None) -> tuple:
        """
        NOTE: I think what we want to do here is get the estimated reward up to a certain
            number of timesteps, and use the return on that path as the reward for each
            member of the group. This way, we can use the same reward for each member of

        Sample a group of actions from the environment at the given stats. If current_state
        is None, use the start state as the current state.
        
        Returns:
            observations: the new observations. Dim [self.group_size, state_dim]
            actions: the sampled actions. Dim [self.group_size, 1] (1 since discrete)
            rewards: the rewards for taking an action. Dim [self.group_size, 1]
            log_probs: the log probs of taking the action. Dim [self.group_size, 1]
        """
        # Duplicate the environment to simulate a single step in the current env
        envs = duplicate_env(env, self.group_size, self.seed)

        # Iterate over the size of the group and get a state, action, reward, 
        # and log prob for each member of the group
        states, actions, rewards, log_probs = [], [], [], []
        for i in range(self.group_size):
            # Sample an action from the policy
            action, old_logprob = self.policy.act(current_state[None], return_log_prob = True)
            assert old_logprob.shape == (1,)
            action, old_logprob = action[0], old_logprob[0]
            # Get the reward and next state from the environment
            state, reward, done, info, _ = envs[i].step(action)

            # Save the action, reward, log prob, and state
            actions.append(action)
            log_probs.append(old_logprob)
            states.append(state)

            # Sample a path!!!! from the policy and get the reward. We will treat this reward
            # as the value of [state] at timestep t
            path_expected_reward = 0
            for _ in range(self.group_ep_len):
                action, _ = self.policy.act(state[None], return_log_prob = True)
                state, reward, done, info, _ = envs[i].step(action)
                path_expected_reward += reward

                if done:
                    break

            rewards.append(path_expected_reward)

        return np.array(states), np.array(actions), np.array(rewards), np.array(log_probs)

    
    def sample_path(self, env: gym.Env, num_episodes=None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym environment

        Returns:
            paths: a list of paths. Each path in paths is a dictionary with
                path["observation"] a numpy array of ordered observations in the path
                path["actions"] a numpy array of the corresponding actions in the path
                path["reward"] a numpy array of the corresponding rewards in the path
            total_rewards: the sum of all rewards encountered during this "path"

        You do not have to implement anything in this function, but you will need to
        understand what it returns, and it is worthwhile to look over the code
        just so you understand how we are taking actions in the environment
        and generating batches to train on.
        """
        episode = 0
        episode_rewards = [] 
        paths = []
        all_group_data = []
        t = 0

        while num_episodes or t < self.config.batch_size:
            state = env.reset()[0]
            states, actions, old_logprobs, rewards = [], [], [], []
            # storing the group data we'll need to calculate advantages and optimize the GRPO loss
            group_states, group_actions, group_log_probs, group_rewards = [], [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                # Note the difference between this line and the corresponding line
                # in PolicyGradient.
                # Sample an action, state, and record a reward for the "regular" path
                action, old_logprob = self.policy.act(states[-1][None], return_log_prob = True)
                assert old_logprob.shape == (1,)
                action, old_logprob = action[0], old_logprob[0]
                # FOR GRPO, we need to sample a group of actions, not just one
                state, reward, done, info, _ = env.step(action)
                actions.append(action)
                old_logprobs.append(old_logprob)
                rewards.append(reward)
                episode_reward += reward

                # Sample a group of actions, states, and rewards for the GRPO loss
                group_state, group_action, group_reward, group_log_prob = self.sample_group(env, state)
                group_states.append(group_state)
                group_actions.append(group_action)
                group_rewards.append(group_reward)
                group_log_probs.append(group_log_prob)

                t += 1
                if done or step == self.config.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            # A single path must contain the info for every group
            path = {  
                # observation shape: [ep_length, state_dim]
                "observation": np.array(states),
                # rewards shape: [ep_length]
                "reward": np.array(rewards),
                # actions shape: [ep_length]
                "action": np.array(actions),
                # log_probs shape: [ep_length]
                "old_logprobs": np.array(old_logprobs),
            }
            group_data = {
                # observations shape: [ep_length, group_size, state_dim]
                "observations" : np.array(group_states),
                # rewards shape: [ep_length, group_size]
                "rewards" : np.array(group_rewards),
                # actions shape: [ep_length, group_size]
                "actions" : np.array(group_actions),
                # log_probs shape: [ep_length, group_size]
                "old_logprobs" : np.array(group_log_probs)
            }
            paths.append(path)
            all_group_data.append(group_data)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards, all_group_data