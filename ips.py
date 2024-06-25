import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import csv
from scipy.integrate import solve_bvp, solve_ivp
from ipydex import IPS, activate_ips_on_exception

from envs.cartpole import (
    CartPoleDiscreteEnv,
    CartPoleContinous2Env,
    CartPoleContinousSwingupEnv,
    CartPoleContinous5StateEnv,
)
from envs.cartpole_transition import (
    CartPoleTransitionDiscreteEnv,
    CartPoleTransitionContinousEnv,
    CartPoleTransitionContinous2Env,
)
from ppo.ppo_agent import PPOAgent
from manual.roly_poly import RolyPolyAgent
from classical.classical_control import *
import util
from callbacks.callback import *
import torch as th

env = CartPoleContinous2Env()
agent = PPOAgent(env)

name = "CartPoleContinous2Env___2023_04_04__11_25_46__best"

agent.load_model(name)


FBA = FeedbackAgent(env, F_LQR_2)
IPS()
