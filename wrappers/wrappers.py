import os, sys
import gym
import gymnasium

from envs.cartpole import CartPoleContinous2Env, CartPoleEnv
from ppo.ppo_agent import PPOAgent

try:
    from imitation.algorithms import bc
except:
    pass


class GymnasiumToGymWrapper(gymnasium.Wrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        if isinstance(self.observation_space, gymnasium.spaces.Box):
            self.observation_space = gym.spaces.Box(low=self.observation_space.low, high=self.observation_space.high)
        if isinstance(self.action_space, gymnasium.spaces.Box):
            self.action_space = gym.spaces.Box(low=self.action_space.low, high=self.action_space.high)

    def reset(self, **kwargs):
        new_obs, trunc = self.env.reset(**kwargs)
        return new_obs

    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        return obs, rew, done, info

    def seed(self, seed):
        return seed


class BCWrapper(bc.BC):
    def __init__(self, observation_space, action_space, demonstrations, rng) -> None:
        # TODO some typechecking
        observation_space = gym.spaces.Box(low=observation_space.low, high=observation_space.high)
        action_space = gym.spaces.Box(low=action_space.low, high=action_space.high)
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            demonstrations=demonstrations,
            rng=rng,
        )
