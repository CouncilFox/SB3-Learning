import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
import os


def main():
    # Define environment name and model path
    environment_name = "CarRacing-v2"  # Replace with your actual environment name
    model_file_name = "PPO_Driving_model_1M"
    PPO_path = os.path.join(
        "Training", "Saved Models", model_file_name
    )  # Replace with the actual path to your model

    # Set up the environment with render_mode
    env_keyword_args = {"render_mode": "human"}
    env = gym.make(environment_name, render_mode="human")

    # Optional: Set render fps
    # env.metadata["render_fps"] = 20

    # Load the model
    model = PPO.load(PPO_path, env)
    print("Model loaded successfully")

    # Evaluate the model and render
    try:
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=10, render=True
        )
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    except Exception as e:
        print(f"Error during evaluation: {e}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    main()
