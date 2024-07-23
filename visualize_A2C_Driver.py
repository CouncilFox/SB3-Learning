import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import os


def main():
    # Define environment name and model path
    environment_name = 'CarRacing-v2'  # Replace with your actual environment name
    model_file_name = "A2C_Driving_model.zip"
    a2c_path = os.path.join(
        "Training", "Saved Models", model_file_name
    )  # Replace with the actual path to your model

    # Set up the environment with render_mode
    env_keyword_args = {"render_mode": "human"}
    env = make_vec_env(
        environment_name, n_envs=1,env_kwargs=env_keyword_args
    )

    # Optional: Set render fps
    # env.metadata["render_fps"] = 20

    # Load the model
    model = A2C.load(a2c_path, env)
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
