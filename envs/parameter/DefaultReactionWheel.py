# Parameters for Environment ReactionWheel
import numpy as np

from envs.reaction_wheel import ReactionWheelEnv

# Seed
START_SEED = 1

# Eval Starting State
EVAL_STATE = [0, 0, 0, 0]
# Cost function for evaluation
Q = np.diag([1000, 1000, 0, 10])
R = 1
# phi_tolerance = 0.01

# action space
max_action = 0.2


# reset bounds
def get_reset_bounds(env):
    # b = 0.05
    b = 0.01
    low = [-0.08, -b, 0, -1]
    high = [0.08, b, 0, 1]
    return low, high


# Rewards
def get_reward(env: ReactionWheelEnv):
    phi, phi_dot, theta, theta_dot = env.state

    truncated = False
    info = {}
    terminated = False
    # terminated = bool(
    #     phi < -env.phi_threshold_radians
    #     or phi > env.phi_threshold_radians
    # )
    # if env.ep_step_count > 1000:
    #     truncated = True
    #     print("reset after 1000 steps")

    # Q = np.diag([1000, 1000, 1000, 1000])
    # R = 1
    # state = np.array(env.state).reshape(3, 1)

    # reward = 1000 - (state.T @ Q @ state + R * env.action**2)[0, 0]

    a = 1
    b = 1

    # reward = a * (1-np.abs(phi))**2 - b*np.abs(phi_dot)**(np.abs(phi))
    reward = a * (1-np.abs(phi))**2 - b*np.abs(phi_dot)**(np.abs(phi)) - 0.001 * env.action

    return reward, terminated, truncated, info


### -------------------------------------------- ###
"""

"""
