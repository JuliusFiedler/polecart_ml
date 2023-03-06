import gymnasium
import sys, os

sys.modules["gym"] = gymnasium
import datetime as dt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ipydex import IPS
from util import *


class PPOAgent:
    def __init__(self, env) -> None:
        self.env = env
        self.model = PPO("MlpPolicy", self.env, verbose=1, seed=env.c.START_SEED)

    def train(self, total_timesteps=300000):
        try:
            self.model.learn(total_timesteps=total_timesteps)
        except KeyboardInterrupt:
            pass
        print("Training Done")
        self.save_model()

    def save_model(self):
        # save model
        t = dt.datetime.now().strftime("_%Y_%m_%d__%H_%M_%S")
        name = self.env.name + "__" + t
        folder_path = os.path.join(ROOT_PATH, "models", name)
        os.makedirs(folder_path)
        model_path = os.path.join(folder_path, "model.h5")
        self.model.save(model_path)
        print(f"Model saved at {model_path}")
        # save metadata
        try:
            para_file_path = self.env.c.__file__
            para_file = None
            with open(para_file_path, "r") as f:
                para_file = f.read()
            metadata_path = os.path.join(folder_path, "metadata.txt")
            with open(metadata_path, "w") as f:
                f.write(para_file)
                f.writelines(
                    [
                        "\n",
                        f"\nTotal Training Steps: {self.env.total_step_count}",
                        f"\nTotal Episode Count: {self.env.episode_count}",
                    ]
                )
        except Exception as e:
            yellow("Parameter file not saved!")
            print(e)
            print(type(e))
            IPS()

    def load_model(self, name):
        path = os.path.join("models", name, "model.h5")
        self.model = PPO.load(path)
        self.model_name = name

    def get_action(self, obs):
        return self.model.predict(obs, deterministic=True)[0]

    def play(self, num_ep=10, render=True):
        self.env.target_change_period = 500
        for i in range(num_ep):
            done = trunc = False
            obs, _ = self.env.reset()
            r_sum = 0
            while not done and not trunc:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, trunc, info = self.env.step(action)
                r_sum += reward
            print("Reward Ep ", i, r_sum)
