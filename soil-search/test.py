import gymnasium as gym
import numpy as np
import time
import os
import soil_search
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

MODEL_ID = "FM_SDF"
MODEL_STEP = 100000


#####################################################################
def main():
    env = gym.make('soil_search/SoilSearchWorld-v0', render_mode="human")

    models_dir = f"soil_search/models/ppo/~model_{MODEL_ID}"
    model_path = f"{models_dir}/{MODEL_STEP}"
    model = PPO.load(model_path)

    obs, info = env.reset()

    done = [False]*4
    import time
    s = 1
    while not np.all(done):
        action, _ = model.predict(obs)
        obs, reward, done, t,  info = env.step(action)

        frame = env.render()
        time.sleep(1)
        print(f"Step {s}, received reward {reward}")
        s+=1



if __name__ == "__main__":
    main()
