# Parameters for Environment CartPole Continuous Swingup
import numpy as np
import util as u

from envs.cartpole import CartPoleEnv

# Seed
START_SEED = 1


# reset bounds
def get_reset_bounds(env):
    b = 0.2
    low = [-b, -b, -np.pi + b, -b]
    high = [b, b, np.pi - b, b]
    return low, high


# Rewards
def get_reward(env: CartPoleEnv):
    x, x_dot, theta, theta_dot = env.state

    truncated = False
    info = {}
    terminated = bool(x < -env.x_threshold or x > env.x_threshold)

    if env.training and env.ep_step_count > 1000:
        terminated = True
        print("reset after 1000 steps")

    # reward_theta = (np.cos(theta) + 1.0) / 2.0
    # reward_x = np.cos((x / env.x_threshold) * (np.pi / 2.0))
    # reward = reward_theta * reward_x

    T = 0.5 * env.masscart * x_dot**2 + 0.5 * env.masspole * (env.length * theta_dot) ** 2
    V = env.masspole * env.gravity * env.length * np.cos(theta)
    E = T + V

    Eopt = env.masspole * env.gravity * env.length

    reward = 1 / (1 + (E - Eopt) ** 2)

    if np.abs(theta) < 0.1:
        reward += 100

    return reward, terminated, truncated, info


### -------------------------------------------- ###
"""Comment Block
energy based approach
"""
