import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Define the environment name
environment_name = "CartPole-v1"

# Create the environment with render mode
env = gym.make(environment_name, render_mode="human")

# Wrap the environment with Monitor
env = Monitor(env)

# Load the pre-trained model
PPO_path = os.path.join("Training", "Saved Models", "best_model")
model = PPO.load(PPO_path, env)

# Evaluate the policy
evaluate_policy(model, env, n_eval_episodes=10, render=True)
