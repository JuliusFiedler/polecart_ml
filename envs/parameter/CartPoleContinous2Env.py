# Parameters for Environment CartPole Continuous
import numpy as np

from envs.cartpole import CartPoleEnv

# Seed
START_SEED = 1


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

    Q = np.diag([1000, 1000, 1000, 1000])
    R = 1
    state = np.array(env.state).reshape(4, 1)

    reward = -(state.T @ Q @ state + R * env.action**2)[0, 0]

    return reward, terminated, truncated, info


### -------------------------------------------- ###
