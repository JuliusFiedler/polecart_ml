import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import csv
from torch import Tensor
import pickle

from envs.cartpole import CartPoleDiscreteEnv, CartPoleContinous2Env, CartPoleContinousSwingupEnv

from ppo.ppo_agent import PPOAgent
from manual.roly_poly import RolyPolyAgent
from classical.classical_control import *
import util
from callbacks.callback import *

from ipydex import IPS

for folder in os.listdir("models"):
    if os.path.isdir(os.path.join("models", folder)):
        print(folder)
        new_path = os.path.join("models", folder, "training_logs.p")
        if os.path.isfile(new_path):
            with open(new_path, "rb") as f:
                log = pickle.load(f)
            if isinstance(log["NN_updates"], dict):
                for k, v in log["NN_updates"]["policy"].items():
                    log["NN_updates"]["policy"][k] = [item.detach().numpy() for item in v]

                with open(new_path, "wb") as f:
                    pickle.dump(log, f)
