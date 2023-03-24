import sys, os
import numpy as np
import gym
import gymnasium
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from wrappers.wrappers import GymnasiumToGymWrapper, BCWrapper
from envs.cartpole import CartPoleContinous2Env, CartPoleContinousSwingupEnv
from ppo.ppo_agent import PPOAgent
from classical.classical_control import *


rng = np.random.default_rng(0)

# og_env = CartPoleContinous2Env()
og_env = CartPoleContinousSwingupEnv()
# og_env.render_mode = "human"
env = GymnasiumToGymWrapper(og_env)



name = "Imitation__CartPoleContinous2Env___2023_03_16__11_03_00_good"
folder_name = os.path.join("models", name)

policy = bc.reconstruct_policy(os.path.join(folder_name, "imi_model.h5"))

env.env.render_mode = "human"
for i in range(10):
    obs = env.reset()
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)
        if done:
            break
