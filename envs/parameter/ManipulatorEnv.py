# Parameters for Environment CartPole Continuous
import numpy as np

from envs.manipulator import ManipulatorEnv

# Seed
START_SEED = 1

# Eval Starting State
EVAL_STATE = [1, 0, np.pi / 180 * 10, 0]
# Cost function for evaluation
Q = np.diag([1000, 1000, 1000, 1000])
R = 1
x_tolerance = 0.05
phi_tolerance = 0.01

# action space
max_action = 10


# reset bounds
def get_reset_bounds(env):
    b = 0.05
    low = [0, 0, 0, 0, 0, 0]
    high = [2*np.pi, 2*np.pi, 0, 0, 2*np.pi, 2*np.pi]
    return low, high


# Rewards
def get_reward(env: ManipulatorEnv):
    phi1, phi2, omega1, omega2, targetphi1, targetphi2 = env.state

    truncated = False
    terminated = False
    info = {}
    if env.ep_step_count > 1000:
        truncated = True
        print("reset after 1000 steps")


    reward = - 10*(phi1 % (2*np.pi) - targetphi1 % (2*np.pi)) ** 2 - (phi2 % (2*np.pi) - targetphi2 % (2*np.pi)) ** 2 - 0.1 * omega1**2 - 0.1 * omega2**2

    if np.abs(phi1 % (2*np.pi) - targetphi1 % (2*np.pi)) < 0.1 and np.abs(phi2 % (2*np.pi) - targetphi2 % (2*np.pi)) % (2*np.pi) < 0.1 and np.abs(omega1)<0.1 and np.abs(omega2) < 0.1:
        terminated = True
    return reward, terminated, truncated, info


### -------------------------------------------- ###
"""

"""
