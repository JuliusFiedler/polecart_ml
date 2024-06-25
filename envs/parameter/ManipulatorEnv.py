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
max_action = 1


# reset bounds
def get_reset_bounds(env):
    b = 0.05
    low = [0, 0, 0, 0]
    high = [2*np.pi, 2*np.pi, 0, 0]
    return low, high


# Rewards
def get_reward(env: ManipulatorEnv):
    phi1, phi2, omega1, omega2, targetx, targety = env.state

    truncated = False
    terminated = False
    info = {}
    if env.ep_step_count > 1000:
        truncated = True
        print("reset after 1000 steps")

    polelength = 1
    join_coord = np.array((np.cos(phi1)*polelength, -np.sin(phi1)*polelength))
    end_coord =  np.array((np.cos(phi1+phi2)*polelength, -np.sin(phi1+phi2)*polelength)) + np.array(join_coord)

    dist = np.sum((np.array([targetx, targety]) - end_coord)**2)
    reward = - dist - 0.01*np.abs(omega1) - 0.01*np.abs(omega2)
    if dist < 0.1 and np.abs(omega1)<0.1 and np.abs(omega2) < 0.1:
        terminated = True
    return reward, terminated, truncated, info


### -------------------------------------------- ###
"""

"""
