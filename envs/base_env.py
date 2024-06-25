"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union
import os
import numpy as np
from scipy.integrate import solve_ivp
from ipydex import IPS

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

import util as u


class BaseEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    

    def __init__(self, render_mode: Optional[str] = None):
        # meta info
        self.set_name()
        self.seed = None
        self.episode_count = 0
        self.ep_step_count = 0
        self.total_step_count = 0
        self.training = False
        self.history = {
            "step": [],
            "episode": [],
            "state": [],
            "action": [],
            "reward": [],
            "terminated": [],
            "truncated": [],
            "info": [],
        }

        self.action_space = None
        self.observation_space = None

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.action = None
        self.reward = None


    def set_name(self):
        self.name = self.__class__.__name__

    def get_force(self, action):
        raise NotImplementedError("This method has to be overwritten by subclass")

    def calc_new_state(self, action):
        raise NotImplementedError

    def step(self, action):
        self.ep_step_count += 1
        self.total_step_count += 1
        if not self.action_space.contains(action):
            action = np.array([action], dtype=float)
        self.action = action
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        self.state = self.calc_new_state(action)

        self.reward, terminated, truncated, info = self.get_reward()

        if self.render_mode == "human":
            self.render()

        self.save_step_data(self.state, action, self.reward, terminated, truncated, info)

        if hasattr(self, "post_processing_state"):
            state = self.post_processing_state()

        return state, self.reward, terminated, truncated, info

    def save_step_data(self, state, action, reward, terminated, truncated, info):
        self.history["step"].append(self.total_step_count)
        self.history["episode"].append(self.episode_count)
        self.history["state"].append(state)
        self.history["action"].append(action)
        self.history["reward"].append(reward)
        self.history["terminated"].append(terminated)
        self.history["truncated"].append(truncated)
        self.history["info"].append(info)

    def get_reward(self):
        raise NotImplementedError

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None and self.seed is not None:
            seed = self.seed + self.episode_count
        self.ep_step_count = 0
        self.episode_count += 1
        super().reset(seed=seed)

    def render(self):
        raise NotImplementedError

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
