# Parameters for Environment CartPole Continuous
import numpy as np

from envs.pendulum import PendulumEnv

# Seed
START_SEED = 1

# Eval Starting State
EVAL_STATE = [np.pi, 0]
# Cost function for evaluation
Q = np.diag([1000, 1000, 1000])
R = 1
x_tolerance = 0.05
phi_tolerance = 0.01

# action space
max_action = 2


# reset bounds
# reset_options = {"x_init": np.pi/180*40, "y_init": 1}


# Rewards
def get_reward(env: PendulumEnv):
    theta, theta_dot = env.state

    
    terminated = False
    # terminated = np.abs(theta) > np.pi/180 * 45
    truncated = False
    info = {}

    if env.ep_step_count > 100:
        truncated = True
        # print("reset after 100 steps")

    if np.abs(theta) < np.pi/2:
        reward = 1 - (10 * theta**2 + theta_dot**2)
    else:
        reward = np.abs(theta_dot)


    return reward, terminated, truncated, info


### -------------------------------------------- ###
"""
swingup
"""
