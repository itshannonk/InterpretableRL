from policy_optimization import PolicyGradient
from config import config_cartpole
import gym
import numpy as np
from network_utils import np2torch
import torch
from general import export_plot
import copy
import time
import tracemalloc

# THIS DUPLICATE FUNCTION WILL DEPEND ON THE ENVIRONMENT
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
        self.eps_clip = config.eps_clip
        self.episode_durations = []

    def normalize_group_rewards(self, rewards: list) -> list:
        """
        Normalize the advantages to have mean 0 and std 1 across each group.

        Args:
            rewards: list of shape [group size]
        
        Returns:
            list of shape [group size]
        """
        rewards = np.array(rewards)
        mean, std = np.mean(rewards), np.std(rewards)
        return ((rewards - mean) / std).tolist() if std > 0 else rewards.tolist()
    
    def unpack_advantages(self, advantages: list, paths: list) -> np.array:
        """
        Unpack the advantages from the list of group advantages to a numpy array
        with shape [batch size]
        """
        unpacked_advantages = []
        for i in range(len(paths)):  # iterate over groups in the batch
            for j in range(len(paths[i])):  # iterate over paths in the group
                # unpack the advantage for that path. The GRPO objective assigns the same advantage
                # for every observation in a path, regardless of time step. So we need to repeat the advantage
                # for each observation in the path
                unpacked_advantages.extend([advantages[i][j]] * len(paths[i][j]["reward"]))
        return np.array(unpacked_advantages)

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
        observations = np2torch(observations)
        actions = np2torch(actions)
        advantages = np2torch(advantages)
        old_logprobs = np2torch(old_logprobs)

        # set up the loss objective using the grpo advantages
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
            if self.config.trace_memory:
                tracemalloc.start()
            start_time = time.perf_counter()

            # collect a minibatch of samples
            paths, total_rewards, norm_advantages = self.sample_path(self.env)
            all_total_rewards.extend(total_rewards)
            advantages = self.unpack_advantages(norm_advantages, paths)

            observations = np.concatenate([np.concatenate([path["observation"] for path in group]) for group in paths])
            actions = np.concatenate([np.concatenate([path["action"] for path in group]) for group in paths])
            old_logprobs = np.concatenate([np.concatenate([path["old_logprobs"] for path in group]) for group in paths])

            # run training operations
            for k in range(self.config.update_freq):
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
        torch.save(self.policy.state_dict(), self.config.model_output)
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
            self.config.plot_duration,
        )
        if self.config.trace_memory:
            export_plot(
                self.peak_memory_usage,
                "Peak memory usage (MB)",
                self.config.env_name,
                self.config.memory_plot,
            )
    
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
        average_group_rewards = [] 
        paths = []
        grpo_advantages = []
        t = 0

        # * self.group_size?
        while num_episodes or t < self.config.batch_size:
            # sample a start state and duplicate it to perform multiple runs
            state = env.reset()  # [0]
            envs = duplicate_env(env, self.group_size, self.seed)
            group_paths = []
            group_rewards = []
            g = 0
            
            # loop through envs to collect sample paths starting from the same start state
            for _env in envs:
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

                    state, reward, done, info = _env.step(env_action)  # sometimes expects 4, sometimes 5?
                    actions.append(action)
                    old_logprobs.append(old_logprob)
                    rewards.append(reward)
                    episode_reward += reward
                    t += 1
                    if done or step == self.config.max_ep_len - 1:
                        group_rewards.append(episode_reward)
                        break
                    if (not num_episodes) and t == self.config.batch_size and g == self.group_size - 1:
                        # the episode ended early, but record the reward anyway
                        group_rewards.append(episode_reward)
                        break

                path = {
                    "observation": np.array(states),
                    "reward": np.array(rewards),
                    "action": np.array(actions),
                    "old_logprobs": np.array(old_logprobs)
                }
                group_paths.append(path)
                g += 1
            
            paths.append(group_paths)
            
            # Normalize the group advantages
            grpo_advantages.append(self.normalize_group_rewards(group_rewards))

            # Use the mean rewards for this group as the reward for this episode
            average_group_rewards.append(sum(group_rewards) / len(group_rewards))
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, average_group_rewards, grpo_advantages