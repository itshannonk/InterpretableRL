import gym
import numpy as np

HUMANOID_ACTION_DIM = 17

class PendulumDiscreteActionWrapper(gym.ActionWrapper):
    """
    A wrapper for the Pendulum environment to convert continuous actions to discrete actions.
    We'll have 3 actions as follows:
        0: -2.0 (apply negative torque)
        1: 0.0  (apply no torque)
        2: 2.0  (apply positive torque)
    """
    def __init__(self, env):
        super().__init__(env)
        self.n_actions = 3
        self.action_space = gym.spaces.Discrete(self.n_actions)

    def action(self, action):
        """
        Convert the discrete action into a continuous action.
        """
        continuous_action = 0.0
        if action == 0:
            continuous_action = -2.0
        elif action == 1:
            continuous_action = 0.0
        elif action == 2:
            continuous_action = 2.0

        return np.array([continuous_action])
    
class HumanoidDiscreteActionWrapper(gym.ActionWrapper):
    """
    A wrapper for the Humanoid environment to convert continuous actions to discrete actions.
    The action space is 17 dimensional, and each dimension can take 5 values:
        0: -0.4 (apply negative torque)
        1: -0.2 (apply less negative torque)
        2: 0.0  (apply no torque)
        3: 0.2  (apply less positive torque)
        4: 0.4  (apply positive torque)
    """
    def __init__(self, env):
        super().__init__(env)
        self.n_bins_per_dim = 5
        self.action_space = gym.spaces.MultiDiscrete([self.n_bins_per_dim] * 17)

    def action(self, action):
        """
        Convert the discrete action into a continuous action.
        """
        continuous_action = np.zeros(HUMANOID_ACTION_DIM)
        for dim, act in enumerate(action):
            if act == 0:
                continuous_action[dim] = -0.4
            elif act == 1:
                continuous_action[dim] = -0.2
            elif act == 2:
                continuous_action[dim] = 0.0
            elif act == 3:
                continuous_action[dim] = 0.2
            elif act == 4:
                continuous_action[dim] = 0.4
        return continuous_action

        
class config:
    env_name: str               # name of the environment
    record: bool                # whether to record the environment
    output_path: str            # path to save the output
    log_path: str               # path to save the log
    scores_output: str          # filename to save the scores
    plot_output: str            # filename to save the plot
    model_output: str           # filename to save the model
    duration_output: str        # filename to save the duration per episode
    plot_duration: str          # filename to save the plot of duration per episode
    record_path: str            # path to save the recorded video   
    record_freq: int            # frequency of recording
    summary_freq: int           # frequency of logging summary statistics
    discretized: bool = False   # whether the action space is originally continuous but has been discretized

    # model and training config
    num_batches: int = 100      # number of batches trained on
    batch_size: int = 2000      # number of steps used to compute each policy update
    max_ep_len: int = 200       # maximum episode length
    learning_rate: int = 3e-2   # learning rate for the policy network

    # parameters for the policy network
    n_layers: int = 1           # number of layers in the policy network
    layer_size: int = 64        # size of each layer in the policy network

    # hyperparameters GRPO and PPO
    eps_clip: float = 0.2       # clip parameter
    update_freq: int = 5        # number of updates per batch

    # hyperparameters for GRPO
    group_size: int = 10        # number of sample paths in each group
    kl_weight: float = 0.03     # weight for KL penalty

    # hyperparameters for PPO
    gamma: float = 0.99         # discount factor for future rewards

    def __init__(self, grpo: bool, env_name: str, seed: int, trace_memory: bool):
        self.env_name = env_name
        self.record = False
        seed_str = "seed=" + str(seed)
        algorithm_type = "grpo" if grpo else "ppo"
        # output config
        self.output_path = "results/{}-{}-{}/".format(
            self.env_name, algorithm_type, seed_str 
        )
        self.model_output = self.output_path + "policy.pth"
        self.log_path = self.output_path + "log.txt"
        self.scores_output = self.output_path + "scores.npy"
        self.plot_output = self.output_path + "scores.png"
        self.duration_output = self.output_path + "duration.npy"
        self.plot_duration = self.output_path + "duration.png"
        self.memory_output = self.output_path + "memory.npy"
        self.memory_plot = self.output_path + "memory.png"
        self.record_path = self.output_path
        self.record_freq = 5
        self.summary_freq = 1

        # since we start new episodes for each batch
        assert self.max_ep_len <= self.batch_size
        if self.max_ep_len < 0:
            self.max_ep_len = self.batch_size

        # whether to trace memory usage
        self.trace_memory = trace_memory

class config_cartpole(config):
    def __init__(self, grpo: bool, seed: int, trace_memory: bool):
        super().__init__(grpo, "CartPole-v1", seed, trace_memory)
        self.max_ep_len = 500

class config_pendulum(config):
    def __init__(self, grpo: bool, seed: int, trace_memory: bool):
        super().__init__(grpo, "Pendulum-v1", seed, trace_memory)
        self.discretized = True

class config_mountaincar(config):
    def __init__(self, grpo: bool, seed: int, trace_memory: bool):
        super().__init__(grpo, "MountainCar-v0", seed, trace_memory)

class config_cheetah(config):
    def __init__(self, grpo: bool, seed: int, trace_memory: bool):
        super().__init__(grpo, "HalfCheetahBulletEnv-v4", seed, trace_memory)

class config_humanoid(config):
    def __init__(self, grpo: bool, seed: int, trace_memory: bool):
        super().__init__(grpo, "Humanoid-v3", seed, trace_memory)
        self.discretized = True
        self.max_ep_len = 1000

def setup_env(env_name: str, grpo: bool, seed: int, trace_memory: bool) -> tuple:
    """
    Return a config object and a gym environment for the given environment name.
    Args:
        env_name: (str) name of the environment
        grpo: (bool) whether to use GRPO or PPO
        seed: (int) random seed for the environment
    """
    if env_name == "CartPole-v1":
        config = config_cartpole(grpo, seed, trace_memory)
    elif env_name == "MountainCar-v0":
        config = config_mountaincar(grpo, seed, trace_memory)
    elif env_name == "Pendulum-v1":
        config = config_pendulum(grpo, seed, trace_memory)
    elif env_name == "HalfCheetahBulletEnv-v0":
        config = config_cheetah(grpo, seed, trace_memory)
    elif env_name == "Humanoid-v3":
        config = config_humanoid(grpo, seed, trace_memory)
    else:
        raise ValueError("Unknown environment name: {}".format(env_name))

    # create the environment
    env = gym.make(env_name)
    env.seed(seed)

    # descretize the action space if needed
    if env_name == "Pendulum-v1":
        env = PendulumDiscreteActionWrapper(env)
    else:
        env = HumanoidDiscreteActionWrapper(env)

    return config, env