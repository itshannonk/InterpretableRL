# 1. Set up the environment
# 1.1. - Install packages
# conda install conda-forge::gym
# conda install conda-forge::shimmy
# conda install conda-forge::stable-baselines3

# 1.2. - Import packages
import gymnasium as gym
from stable_baselines3 import DQN
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import time
from gymnasium.wrappers import RecordVideo
import os

# 2. Problem Setup
# 2.2. - Defining the RL Task
# Initialize the environment
env = gym.make('CartPole-v1', render_mode="human")

# Print state and action spaces for clarity
print(f"State space: {env.observation_space}")
print(f"Action space: {env.action_space}")

# 3. Training a Standard RL Model
# 3.1 Baseline RL Algorithm
# Create the environment
env = gym.make('CartPole-v1')
env = RecordVideo(env, video_folder=".", episode_trigger=lambda x: True)

# # Initialize the model with a basic MLP policy
# model = DQN('MlpPolicy', env, verbose=1)

# # Train the model
# model.learn(total_timesteps=50000)

# # Save the model
# model.save('dqn_cartpole')

# Load the model
model = DQN.load('dqn_cartpole')

# 3.2. Evaluating the baseline
# Test the trained model
state = np.array(env.reset()[0])  # converted to np array
done = False

while not done:
    # Predict the action for the current state
    action, _states = model.predict(state, deterministic=True)
    
    # Take the action and get the next state
    state, reward, done, info, _ = env.step(action)  # added _
    
    # Render the environment to visualize performance
    time.sleep(0.01)
    env.render()
    print("render step done")

env.close()
print("Video has been saved")

# 4. Decision Tree RL: Theory meets practice
