# Parameters for Environment ReactionWheel
import numpy as np

from envs.reaction_wheel import ReactionWheelEnv

# Seed
START_SEED = 1

# Eval Starting State
EVAL_STATE = [1*np.pi/180, 0, 0]
# Cost function for evaluation
# Q = np.diag([1000, 1000, 1000])
# R = 1
# phi_tolerance = 0.01

# action space
max_action = 1e-3


# reset bounds
def get_reset_bounds(env):
    b = 0.05
    low = [-0.1, -b, -b]
    high = [0.1, b, b]
    return low, high


# Rewards
def get_reward(env: ReactionWheelEnv):
    phi, phi_dot, theta_dot = env.state

    truncated = False
    info = {}
    terminated = bool(
        phi < -env.phi_threshold_radians
        or phi > env.phi_threshold_radians
    )
    if env.ep_step_count > 1000:
        truncated = True
        print("reset after 1000 steps")

    # Q = np.diag([1000, 1000, 1000, 1000])
    # R = 1
    # state = np.array(env.state).reshape(3, 1)

    # reward = 1000 - (state.T @ Q @ state + R * env.action**2)[0, 0]

    reward = 1

    return reward, terminated, truncated, info


### -------------------------------------------- ###
"""
R = 1000 - J_LQR
no cap
"""
