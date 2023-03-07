# Parameters for Environment CartPole Continuous

# Seed
START_SEED = 1

# reset bounds
def get_reset_bounds(env):
    b = 0.05
    low = [-0.5, -b, -b, -b]
    high = [0.5, b, b, b]
    return low, high

# Rewards
REW_STEP = 1
REW_FACTOR_DELTA_X_SQARED = -1

### -------------------------------------------- ###