from policy_optimization import PolicyGradient
from config import config_cartpole
import gym
import numpy as np
from network_utils import np2torch
import torch
from general import export_plot
from baseline_network import BaselineNetwork
import time
import tracemalloc

class PPO(PolicyGradient):
    def __init__(self, env: gym.Env, config: config_cartpole, seed: int, logger=None):
        super(PPO, self).__init__(env, config, seed, logger)
        self.baseline_network = BaselineNetwork(env, config)
        self.eps_clip = config.eps_clip
        self.episode_durations = []

    def get_returns(self, paths):
        """
        Calculate the returns G_t for each timestep

        Args:
            paths: recorded sample paths. See sample_path() for details.

        Return:
            returns: return G_t for each timestep

        After acting in the environment, we record the observations, actions, and
        rewards. To get the advantages that we need for the policy update, we have
        to convert the rewards into returns, G_t, which are themselves an estimate
        of Q^π (s_t, a_t):

           G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T

        where T is the last timestep of the episode.

        Note that here we are creating a list of returns for each path

        TODO: compute and return G_t for each timestep. Use self.config.gamma.
        """

        all_returns = []
        for path in paths:
            rewards = path["reward"]
            #######################################################
            #########   YOUR CODE HERE - 5-10 lines.   ############
            returns = []
            current_g = 0
            for reward in rewards[::-1]:
                current_g = reward + self.config.gamma * current_g
                returns.append(current_g)
            returns = np.array(returns[::-1])
            #######################################################
            #########          END YOUR CODE.          ############
            all_returns.append(returns)
        returns = np.concatenate(all_returns)

        return returns
    
    def normalize_advantage(self, advantages):
        """
        Args:
            advantages: np.array of shape [batch size]
        Returns:
            normalized_advantages: np.array of shape [batch size]

        TODO:
        Normalize the advantages so that they have a mean of 0 and standard
        deviation of 1. Put the result in a variable called
        normalized_advantages (which will be returned).

        Note:
        This function is called only if self.config.normalize_advantage is True.
        """
        #######################################################
        #########   YOUR CODE HERE - 1-2 lines.    ############
        mean, std = np.mean(advantages), np.std(advantages)
        normalized_advantages = (advantages - mean) / std
        #######################################################
        #########          END YOUR CODE.          ############
        return normalized_advantages
    
    def calculate_advantage(self, returns, observations):
        """
        Calculates the advantage for each of the observations
        Args:
            returns: np.array of shape [batch size]
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        if True:  # self.config.use_baseline:
            # override the behavior of advantage by subtracting baseline
            advantages = self.baseline_network.calculate_advantage(
                returns, observations
            )
        else:
            advantages = returns

        if True:  # self.config.normalize_advantage:
            advantages = self.normalize_advantage(advantages)

        return advantages
    

    def update_policy(self, observations: np.array, actions: np.array, advantages: np.array, old_logprobs: np.array):
        """        
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
            actions: np.array of shape
                [batch size, dim(action space)] if continuous
                [batch size] (and integer type) if discrete
            advantages: np.array of shape [batch size, 1]
            old_logprobs: np.array of shape [batch size]

        Perform one update on the policy using the provided data using the PPO clipped
        objective function.
        """
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)

        # set up the ppo loss objective
        new_logprobs = self.policy.action_distribution(observations).log_prob(actions)
        prob_ratio = torch.exp(new_logprobs - old_logprobs)
        clipped = torch.clamp(prob_ratio, 1 - self.eps_clip, 1 + self.eps_clip)
        loss = -torch.min(prob_ratio * advantages, clipped * advantages).mean()

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
            # start timer and memory tracking
            start_time = time.perf_counter()
            if self.config.trace_memory:
                tracemalloc.start()

            # collect a minibatch of samples
            paths, total_rewards = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            old_logprobs = np.concatenate([path["old_logprobs"] for path in paths])

            # compute Q-val estimates (discounted future returns) for each time step
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            # run training operations
            for k in range(self.config.update_freq):
                self.baseline_network.update_baseline(returns, observations)
                self.update_policy(observations, actions, advantages, 
                                   old_logprobs)

            # end timer and memory tracking
            end_time = time.perf_counter()
            self.episode_durations.append(end_time - start_time)
            if self.config.trace_memory:
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self.peak_memory_usage.append(peak / 10**6)  # convert to MB

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
        msg = "[FINAL] Average duration: {:04.2f}s".format(np.mean(self.episode_durations))
        self.logger.info(msg)
        if self.config.trace_memory:
            msg = "[FINAL] Average peak memory usage: {:04.2f}MB".format(np.mean(self.peak_memory_usage))
            self.logger.info(msg)

        # Save the resulting policy and the training statistics
        torch.save(self.policy.state_dict(), self.config.output_path + "/policy.pth")
        np.save(self.config.scores_output, averaged_total_rewards)
        np.save(self.config.duration_output, self.episode_durations)
        if self.config.trace_memory:
            np.save(self.config.memory_output, self.peak_memory_usage)

        # Plot the training stats
        export_plot(
            averaged_total_rewards,
            "Score",
            self.config.env_name,
            self.config.plot_output,
        )
        export_plot(
            self.episode_durations,
            "Episode durations (s)",
            self.config.env_name,
            self.config.record_path + "durations.png",
        )
        if self.config.trace_memory:
            export_plot(
                self.peak_memory_usage,
                "Peak memory usage (MB)",
                self.config.env_name,
                self.config.memory_plot,
            )

    def sample_path(self, env, num_episodes=None):
        """
        Sample paths (trajectories) from the environment.

        Args:
            num_episodes: the number of episodes to be sampled
                if none, sample one batch (size indicated by config file)
            env: open AI Gym envinronment

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
        t = 0

        while num_episodes or t < self.config.batch_size:
            state = env.reset()  # [0]
            states, actions, old_logprobs, rewards = [], [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                # Note the difference between this line and the corresponding line
                # in PolicyGradient.
                action, old_logprob = self.policy.act(states[-1][None], return_log_prob = True)
                assert old_logprob.shape == (1,)
                action, old_logprob = action[0], old_logprob[0]

                env_action = action
                # map the discrete action to the environment's original continuous action space
                if self.discretized:
                    env_action = env.action(env_action)

                state, reward, done, info = env.step(action)  # might need to unpack 5?
                actions.append(action)
                old_logprobs.append(old_logprob)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done or step == self.config.max_ep_len - 1:
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            path = {
                "observation": np.array(states),
                "reward": np.array(rewards),
                "action": np.array(actions),
                "old_logprobs": np.array(old_logprobs)
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards