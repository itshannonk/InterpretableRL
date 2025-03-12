import numpy as np
import gym
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import KBinsDiscretizer
import joblib
import torch
import torch.nn as nn
from policy import CategoricalPolicy, GaussianPolicy
from config import PendulumDiscreteActionWrapper
import matplotlib.pyplot as plt

DISCRETIZED = False

def generate_dataset(env, policy, episodes: int = 1000, max_ep_len: int = 200):
    """
    Generate a dataset of observations and actions using the given model. Additionally, keep
    track the rewards obtained during the episode. To evaluate the policy network model.
    
    Args:
        env: The environment to sample from.
        policy: The policy network used to generate actions.
        n_samples: The number of samples to generate.
        
    Returns:
        observations: A numpy array of shape (n_samples, observation_dim).
        actions: A numpy array of shape (n_samples, action_dim).
    """
    # Generate samples
    with torch.no_grad():
        observations = []
        actions = []
        rewards = []
        
        for _ in range(episodes):
            obs = env.reset()
            done = False
            reward = 0
            t = 0
            while not done and t < max_ep_len:
                # BasePolicy.act takes multiple observations and returns multiple actions
                action = policy.act(np.array([obs]))[-1]
                observations.append(obs)
                actions.append(action)

                if DISCRETIZED:
                    action = env.action(action)

                obs, r, done, _ = env.step(action)
                reward += r
                t += 1
            rewards.append(reward)
        return np.array(observations), np.array(actions), sum(rewards) / len(rewards)

def train_decision_tree(observations: np.array, actions: np.array, seed: int, max_depth: int = 5, discretize_actions: bool = False, num_bins: int = 10) -> DecisionTreeClassifier:
    """
    Train a decision tree model on the given observations and actions.
    
    Args:
        observations: A numpy array of shape (n_samples, observation_dim).
        actions: A numpy array of shape (n_samples, action_dim).
        
    Returns:
        model: The trained decision tree model.
    """
    if discretize_actions:
        print("Discretizing the actions..")
        discretizer = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='uniform')
        discretized_actions = discretizer.fit_transform(actions)
    
        # Ensure the shape of discretized actions matches the number of observations
        discretized_actions = discretized_actions.reshape(-1, actions.shape[1])
        
        decision_tree = DecisionTreeClassifier(random_state=seed, max_depth=max_depth)
        decision_tree.fit(observations, discretized_actions)
        
        return decision_tree
    else:
        return DecisionTreeClassifier(random_state=seed, max_depth=max_depth).fit(observations, actions)

def evaluate_decision_tree(env, model: DecisionTreeClassifier, episodes: int, max_ep_length: int) -> float:
    """
    Evaluate the decision tree model on the given observations.
    
    Args:
        model: The trained decision tree model.
        observations: A numpy array of shape (n_samples, observation_dim).
        
    Returns:
        average_reward: The average reward obtained by the model.
    """
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        reward = 0
        t = 0
        while not done and t < max_ep_length:
            action = model.predict(state.reshape(1, -1))[-1]

            if DISCRETIZED:
                action = env.action(action)

            state, r, done, _ = env.step(action)
            reward += r
            t += 1
        rewards.append(reward)
    return sum(rewards) / episodes

def save_decision_tree(model: DecisionTreeClassifier, filename: str):
    """
    Save the decision tree model to a file.
    
    Args:
        model: The trained decision tree model.
        filename: The name of the file to save the model to.
    """
    joblib.dump(model, filename)

def plot_rewards(rewards: list, depths: list, neural_net_reward: float, env_name: str, filename: str):
    """
    Plot the rewards obtained by the decision tree models for different depths.
    
    Args:
        rewards: A list of average rewards obtained by the decision tree models.
        depths: A list of the decision tree depths.
        neural_net_reward: The average reward obtained by the neural network model.
        env_name: The name of the environment.
        filename: The name of the file to save the plot to.
    """
    plt.plot(depths, rewards, label="Decision Tree Depth")
    plt.axhline(y=neural_net_reward, color='r', linestyle='--', label='Avg. NN Reward')
    plt.xlabel("Decision Tree Depth")
    plt.ylabel("Average Reward")
    plt.title(f"Decision Tree Depth vs Average Reward - {env_name}")

    plt.legend()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    # TODO: this file should accept parameters
    # - env_name
    # - seed
    # - path to model directory (we'll load the model and save the decision tree here)
    # - model architecture
    # - number of episodes to generate
    # - maximum episode length
    # - decision tree depth
    env_name = "Humanoid-v3"
    seed = 3
    model_dir = "results/Humanoid-v3-grpo-seed=1/"
    num_episodes = 1000
    max_ep_len = 500
    # depth = 5

    if env_name == "Pendulum-v1":
        DISCRETIZED = True

    # Create the environment
    env = gym.make(env_name)
    if env_name == "Pendulum-v1":
        env = PendulumDiscreteActionWrapper(env)

    env.seed(seed)

    # Load the model
    # Define the model architecture
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    action_dim = env.action_space.n if discrete else env.action_space.shape[0]
    observation_dim = env.observation_space.shape[0]

    input_size, size, output_size, n_layers = observation_dim, 64, action_dim, 2
    layers = []
    layers.extend([nn.Linear(input_size, size), nn.ReLU()])
    layers.extend([nn.Linear(size, size), nn.ReLU()] * (n_layers - 1))
    layers.append(nn.Linear(size, output_size))
    model = nn.Sequential(*layers)

    # load from file
    state_dict = torch.load(model_dir + "policy.pth")
    # remove the 'network.' prefix from the keys
    state_dict = {k.replace("network.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Create the policy
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    policy = CategoricalPolicy(model) if discrete else GaussianPolicy(model, action_dim)

    # Generate a dataset from the policy network
    print("Generating dataset...")
    observations, actions, average_nn_reward = generate_dataset(env, policy, num_episodes, max_ep_len)
    print(f"Done.\nAverage reward for policy network model: {average_nn_reward}")

    # Run the experiment for multiple decision tree depths
    max_depth_power = 7  # 2^7 = 128 leaves
    average_rewards = []
    for i in range(1, max_depth_power + 1):
        depth = i
        # Train a decision tree using the policy network dataset
        print(f"Training decision tree with depth {depth}...")
        decision_tree_model = train_decision_tree(observations, actions, seed, depth, discretize_actions=env_name == "Humanoid-v3")

        # Evaluate the decision tree policy
        print("Done.\nEvaluating decision tree...")
        average_reward = evaluate_decision_tree(env, decision_tree_model, num_episodes, max_ep_len)
        average_rewards.append(average_reward)
        print(f"Average reward for decision tree model: {average_reward}")

        # Save the decision tree model
        print("Saving decision tree model...")
        model_save_path = model_dir + "decision_tree_model_depth-" + str(depth) + ".pkl"
        save_decision_tree(decision_tree_model, model_save_path)
        print("Done.")

    # Plot the decision tree rewards for the powers of 2 and show the average reward for the
    # neural net model as a baseline
    print("Plotting results...")
    plot_rewards(average_rewards, list(range(1, max_depth_power + 1)), average_nn_reward, env_name,
                  model_dir + "decision_tree_rewards.png")
    
    # Save the list of average rewards for this seed
    print("Saving rewards...")
    np.save(model_dir + "decision_tree_rewards.npy", average_rewards)

    env.close()
