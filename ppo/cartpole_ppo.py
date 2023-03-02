import gymnasium
import sys, os
sys.modules["gym"] = gymnasium
import datetime as dt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ipydex import IPS


class PPOAgent():
    def __init__(self, env) -> None:
        self.env = env
        self.model = PPO("MlpPolicy", self.env, verbose=1)
        
    def train(self, total_timesteps=300000):
        try:
            self.model.learn(total_timesteps=total_timesteps)
        except KeyboardInterrupt:
            pass
        t = dt.datetime.now().strftime("_%Y_%m_%d__%H_%M_%S")
        name = "cartpole_model__" + self.env.name + "__" + t + ".h5"
        self.model.save(os.path.join("ppo", name))
        print("Training Done")
        print(f"Model saved as {name}")

    def load_model(self, name):
        path = os.path.join("ppo", name)
        self.model = PPO.load(path)
        
    def get_action(self, obs):
        return self.model.predict(obs, deterministic=True)[0]
    
    def play(self, num_ep=10, render=True):
        self.env.target_change_period = 500
        for i in range(num_ep):
            done = False
            obs, _ = self.env.reset()
            r_sum = 0
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.env.step(action)
                r_sum += reward
            print("Reward Ep ", i, r_sum)