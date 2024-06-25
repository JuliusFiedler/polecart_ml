# Parameters for Environment ReactionWheel
import numpy as np

from envs.ballbeam import BallBeamEnv

# Seed
START_SEED = 1

# Eval Starting State
EVAL_STATE = [0, 0, 0, 0]
# Cost function for evaluation
Q = np.diag([1000, 1000, 0, 10])
R = 1
# phi_tolerance = 0.01

# action space
max_action = 10


# reset bounds
def get_reset_bounds(env):
    # b = 0.05
    b = 0.01
    low = [-1, -b, -0.1, 0]
    high = [1, b, 0.1, 0]
    return low, high


# Rewards
def get_reward(env: BallBeamEnv):
    r, r_dot, theta, theta_dot = env.state

    truncated = False
    info = {}
    terminated = False
    # terminated = bool(
    #     theta < -env.theta_threshold_radians
    #     or theta > env.theta_threshold_radians
    #     or r > env.r_threshold
    #     or r < - env.r_threshold
    # )
    # if env.ep_step_count > 1000:
    #     truncated = True
    #     print("reset after 1000 steps")

    # Q = np.diag([1000, 1000, 1000, 1000])
    # R = 1
    # state = np.array(env.state).reshape(3, 1)

    # reward = 1000 - (state.T @ Q @ state + R * env.action**2)[0, 0]


    reward = 1 - np.abs(r)**2

    return reward, terminated, truncated, info


### -------------------------------------------- ###
"""

"""
