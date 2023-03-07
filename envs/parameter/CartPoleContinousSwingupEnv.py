# Parameters for Environment CartPole Continuous Swingup
import numpy as np

# Seed
START_SEED = 1


# reset bounds
def get_reset_bounds(env):
    b = 0.05
    low = [-b, -b, -b, -b]
    high = [b, b, b, b]
    # low = [-b, -b, -np.pi, -b]
    # high = [b, b, -np.pi, b]
    return low, high


# Rewards
def get_reward(env):
    x, x_dot, theta, theta_dot = env.state

    truncated = False
    info = {}
    terminated = bool(x < -env.x_threshold or x > env.x_threshold)

    if env.training and env.ep_step_count > 500:
        terminated = True

    reward = np.cos(theta)

    return reward, terminated, truncated, info


### -------------------------------------------- ###
