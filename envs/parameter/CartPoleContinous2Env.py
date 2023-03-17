# Parameters for Environment CartPole Continuous
import numpy as np

from envs.cartpole import CartPoleEnv

# Seed
START_SEED = 1

# Eval Starting State
EVAL_STATE = [1, 0, np.pi / 180 * 10, 0]


# reset bounds
def get_reset_bounds(env):
    b = 0.05
    low = [-0.5, -b, -b, -b]
    high = [0.5, b, b, b]
    return low, high


# Rewards
def get_reward(env: CartPoleEnv):
    x, x_dot, theta, theta_dot = env.state

    truncated = False
    info = {}
    terminated = bool(
        x < -env.x_threshold
        or x > env.x_threshold
        or theta < -env.theta_threshold_radians
        or theta > env.theta_threshold_radians
    )
    if env.ep_step_count > 1000:
        terminated = True
        print("reset after 1000 steps")

    # Q = np.diag([1000, 1000, 1000, 1000])
    # R = 1
    # state = np.array(env.state).reshape(4, 1)

    # reward = 1 / (1 + (state.T @ Q @ state + R * env.action**2)[0, 0])

    reward = 1 - (x) ** 2

    return reward, terminated, truncated, info


### -------------------------------------------- ###
"""
standard approach, 1 per step - x**2
"""
