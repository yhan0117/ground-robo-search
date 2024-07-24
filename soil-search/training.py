import gymnasium as gym
import soil_search
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

import time, os
import numpy as np

MODEL_ID = "FM_SDF"
CUR_MODEL_STEP = 30000
NUM_STEP = 70000

def main():
    # Parallel environments
    vec_env = SubprocVecEnv([lambda: gym.make('soil_search/SoilSearchWorld-v0')])
    models_dir = f"soil_search/models/ppo/~model_{MODEL_ID}"
    log_dir = f"soil_search/models/ppo/logs"
    log_name = f"{MODEL_ID}"

    if not os.path.isdir(models_dir): os.makedirs(models_dir)
    if not os.path.isdir(log_dir): os.makedirs(log_dir)

    if CUR_MODEL_STEP is None:
        model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=log_dir, ent_coef=0.02)
    else:
        try: 
            model = PPO.load(f"{models_dir}/{CUR_MODEL_STEP}", env=vec_env, verbose=1)
        except:
            print("Model does not exist, check if model step is correct.")
            exit()
    
    for steps in range(CUR_MODEL_STEP, CUR_MODEL_STEP+NUM_STEP,5000):
        model.learn(total_timesteps=5000, reset_num_timesteps=False, tb_log_name=log_name)
        model.save(f"{models_dir}/{steps+5000}")
        print(f"Model saved to {models_dir}/{steps+5000}")


if __name__ == "__main__":
    main()

