import numpy as np
import csv
import os

import util


class FeedbackAgent:
    def __init__(self, env, F) -> None:
        self.env = env
        self.F = F
        self.model_name = F["note"]

    def get_action(self, state):
        state = util.project_to_interval(state - self.F["eq"], min=-np.pi, max=np.pi)
        u = -self.F["F"] @ np.array(state)
        action = np.clip(u, self.env.action_space.low[0], self.env.action_space.high[0])
        return action


class FeedforwardAgent:
    def __init__(self, env, path):
        self.env = env

        if os.path.isabs(path):
            self.path = path
        else:
            self.path = os.path.join(util.ROOT_PATH, "trajectories", path)
        with open(self.path, newline="") as csvfile:
            self.actions = []
            reader = csv.reader(csvfile, delimiter=",")
            for row in reader:
                self.actions.append(row[0])

        self.counter = 0
        self.trajectory_end = False

    def get_action(self, obs):
        try:
            a = np.array([self.actions[self.counter]], dtype=float)
            a = np.clip(a, self.env.action_space.low[0], self.env.action_space.high[0])
        except IndexError as e:
            a = np.clip(np.array([0]), self.env.action_space.low[0], self.env.action_space.high[0])
            util.yellow(f"WARNING: Feedforward Controller at end of trajectory. Output will be 0 from now on.")
            self.trajectory_end = True
        self.counter += 1
        return a

    def reset_trajectory(self):
        self.counter = 0
        self.trajectory_end = False


F_EV_imag_1 = {
    "note": "place eigenvalues: [-1.5+a*1j, -1.5-a*1j, -1.3 + b*1j, -1.3 - b*1j]",
    "F": np.array([[-2.4919724770642167, -2.514984709480122, -23.425986238532126, -4.057492354740066]]),
    "eq": np.array([0, 0, 0, 0]),
}
F_EV_real_1 = {
    "note": "place eigenvalues: [-8., -7., -6., -5.]",
    "F": np.array([[-85.62691131487867, -54.33231396529075, -178.12345565736146, -40.166156982641404]]),
    "eq": np.array([0, 0, 0, 0]),
}
F_LQR_1 = {
    "note": "LQR Q=np.diag([100, 100, 100, 100]) R=1",
    "F": np.array([[-9.999999999999986, -16.266056714636235, -90.5387468119659, -22.64298172774513]]),
    "eq": np.array([0, 0, 0, 0]),
}
F_LQR_2 = {
    "note": "LQR Q=np.diag([1000, 1000, 1000, 1000]) R=1",
    "F": np.array([[-31.622776601683427, -50.259303062636604, -246.50705065047896, -63.60946003330605]]),
    "eq": np.array([0, 0, 0, 0]),
}
F_LQR_3 = {
    "note": "LQR Q=np.diag([1000, 1000, 0, 0]) R=1",
    "F": np.array([[-31.622776601684222, -23.899312329610922, -98.40512423898201, -21.037765755811748]]),
    "eq": np.array([0, 0, 0, 0]),
}
F_LQR_4 = {
    "note": "LQR Q=np.diag([100, 100, 0, 0]) R=100",
    "F": np.array([[-0.9999999999998335, -1.8680009386439562, -26.925641920732488, -6.074588171203054]]),
    "eq": np.array([0, 0, 0, 0]),
}
F_EV_real_LOWER_EQ_1 = {
    "note": "stabilizing the lower equilibrium point, place eigenvalues: [-8., -7., -6., -5.]",
    "F": np.array([[85.62691131499069, 54.332313965344916, 72.8765443425107, -14.166156982672117]]),
    "eq": np.array([0, 0, -np.pi, 0]),
}
F_EV_real_LOWER_EQ_2 = {
    "note": "stabilizing the lower eq point, place eigenvalues: [-1.5+a*1j, -1.5-a*1j, -1.3 + b*1j, -1.3 - b*1j]",
    "F": np.array([[2.491972477064221, 2.514984709480121, 1.314013761467891, 1.5425076452599387]]),
    "eq": np.array([0, 0, -np.pi, 0]),
}
F_LQR_LOWER_EQ_1 = {
    "note": "LQR stabilizing the lower equilibrium point Q=np.diag([1000, 1000, 1000, 1000]) R=1",
    "F": np.array([[31.622776601683597, 43.28803503081849, 125.73324499512717, 15.65939564338845]]),
    "eq": np.array([0, 0, -np.pi, 0]),
}
F_NN_lin_1 = {
    "note": "NN cartpole_model__CartPoleContinous2Env___2023_03_02__16_40_49: linearized",
    "F": -np.array([[16.947, 31.633, 150.457, 40.509]]),
    "eq": np.array([0, 0, 0, 0]),
}
