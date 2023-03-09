# Parameters for Environment CartPole Continuous Swingup
import numpy as np
import util as u

# Seed
START_SEED = 1


# reset bounds
def get_reset_bounds(env):
    b = 0.2
    low = [-b, -b, -np.pi + b, -b]
    high = [b, b, np.pi - b, b]
    return low, high


# Rewards
def get_reward(env):
    x, x_dot, theta, theta_dot = env.state

    truncated = False
    info = {}
    terminated = bool(x < -env.x_threshold or x > env.x_threshold)

    if env.training and env.ep_step_count > 1000:
        terminated = True
        print("reset after 1000 steps")

    # reward = 10.0 - 10.0 * np.abs(u.project_to_interval(theta)) - 0.01 * np.abs(x)
    reward_theta = (np.cos(theta) + 1.0) / 2.0
    reward_x = np.cos((x / env.x_threshold) * (np.pi / 2.0))

    reward = reward_theta * reward_x

    return reward, terminated, truncated, info


### -------------------------------------------- ###
"""Comment Block
source: https://codesuche.com/view-source/python/google/brain-tokyo-workshop/learntopredict/cartpole/cartpole_swingup.py/
"""
