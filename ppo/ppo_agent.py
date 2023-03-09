import gymnasium
import sys, os
import datetime

sys.modules["gym"] = gymnasium
import datetime as dt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from ipydex import IPS

from util import *


class PPOAgent:
    def __init__(self, env) -> None:
        self.env = env
        self.seed = env.c.START_SEED
        self.model = None
        self.tensorboard_log = None

    def create_model(self):
        self.model = PPO("MlpPolicy", self.env, verbose=1, seed=self.seed, tensorboard_log=self.tensorboard_log)

    def train(self, total_timesteps=300000, callback=None, save_model=True):
        self.env.training = True
        # Create Folders and setup logs
        if save_model:
            t = dt.datetime.now().strftime("_%Y_%m_%d__%H_%M_%S")
            name = self.env.name + "__" + t
            self.folder_path = os.path.join(ROOT_PATH, "models", name)
            os.makedirs(self.folder_path)

            self.env = Monitor(self.env, filename=self.folder_path)

            self.tensorboard_log = self.folder_path

            if callback is not None:
                callback.log_path = self.folder_path

            # save metadata, pre training!
            para_file_path = self.env.c.__file__
            para_file = None
            with open(para_file_path, "r") as f:
                para_file = f.read()
            metadata_path = os.path.join(self.folder_path, "metadata.txt")
            with open(metadata_path, "w") as f:
                f.write(para_file)

        # create Model
        if self.model is None:
            self.create_model()
        # Training
        t1 = datetime.datetime.now()
        try:
            self.model.learn(total_timesteps=total_timesteps, callback=callback)
        except KeyboardInterrupt:
            pass
        t2 = datetime.datetime.now()
        self.training_time = smooth_timedelta(t1, t2)
        print("Training Done")
        # Save Model data
        if save_model:
            self.save_model()
        self.env.training = False

    def save_model(self):
        # save model
        model_path = os.path.join(self.folder_path, "model.h5")
        self.model.save(model_path)
        print(f"Model saved at {model_path}")
        # save metadata
        try:
            # para_file_path = self.env.c.__file__
            # para_file = None
            # with open(para_file_path, "r") as f:
            #     para_file = f.read()
            metadata_path = os.path.join(self.folder_path, "metadata.txt")
            with open(metadata_path, "a") as f:
                # f.write(para_file)
                f.writelines(
                    [
                        "\n",
                        f"\nTotal Training Steps: {self.env.total_step_count}",
                        f"\nTotal Episode Count: {self.env.episode_count}",
                        f"\nTraining Duration: {self.training_time}",
                    ]
                )
        except Exception as e:
            yellow("Parameter file not saved!")
            print(e)
            print(type(e))
            IPS()

    def load_model(self, name):
        if self.model is None:
            self.create_model()
        path = os.path.join("models", name, "model.h5")
        # self.model = PPO.load(path)
        self.model.set_parameters(path)
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
