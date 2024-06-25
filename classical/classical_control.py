import numpy as np
from numpy import sin, cos
import csv
import os, sys
import matplotlib.pyplot as plt
from ipydex import IPS

import util
from agents.agent import BaseAgent


class FeedbackAgent(BaseAgent):
    def __init__(self, env, F) -> None:
        self.env = env
        self.F = F
        self.model_name = self.env.name + "__" + F["note"]

    def get_action(self, state, *args):
        state = util.project_to_interval(state - self.F["eq"], min=-np.pi, max=np.pi)
        u = -self.F["F"] @ np.array(state).T
        action = np.clip(u, self.env.action_space.low[0], self.env.action_space.high[0])
        return action.T

    def get_cost(self, state, action=0):
        if state.shape[1] == self.F["F"].shape[1]:
            J = np.diag(state @ self.F["Q"] @ state.T)
        else:
            J = state.T @ self.F["Q"] @ state

        return J

    def get_value(self, state, action=0):
        return self.get_cost(state, action)

    def load_model(self, name):
        print(util.yellow("This is a feedback Agent, loading doesnt make sense here."))

    def eval(self):
        self.run_eval_episodes()

class ExactLinearizationAgent:
    """for reaction wheel pendulum"""
    def __init__(self, env) -> None:
        self.env = env
        self.model_name = "ELA"
   
    def get_action(self, state, *args):
        p1, pdot1, p2, pdot2 = state
        g, m1, m2, s, l, I1, I2, J, m = self.env.parameters
        k1 = 1
        k2 = 3
        k3 = 3
        u = -(I1 + l**2*m2 + m1*s**2)*(-g*k2*(l*m2 + m1*s)*sin(p1) - g*k3*pdot1*(l*m2 + m1*s)*cos(p1) - g*(l*m2 + m1*s)*(-g*(l*m2 + m1*s)*cos(p1) + pdot1**2*(I1 + l**2*m2 + m1*s**2))*sin(p1)/(-I1 - l**2*m2 - m1*s**2) - k1*(I2*pdot2 + pdot1*(I1 + I2 + l**2*m2 + m1*s**2)))/(g*(l*m2 + m1*s)*cos(p1))
        return np.clip(u, self.env.action_space.low[0], self.env.action_space.high[0])

    def load_model(self, name):
        pass
    
class JacobianApproximationControl:
    """for beam ball"""
    def __init__(self, env) -> None:
        self.env = env
        poles = 2
        self.ampl = 3
        self.freq = np.pi/5
        self.BG = env.g* env.B
        self.gain = [util.binom(4, i) * poles**(4-i) for i in range(4)]
        
    def get_action(self, state, t):
        x1, x2, x3, x4 = state
        y = [x1, x2, -self.BG*x3, -self.BG*x4]
        yref = [self.ampl * util.dsin(i, self.freq, t) for i in range(5)]

        u = 1./self.BG * sum([self.gain[i]*(y[i]-yref[i]) for i in range(4)], start=-yref[4])
        
        tau = self.env.u_to_tau(u, state)
        return tau

class FeedforwardAgent:
    def __init__(self, env, path):
        self.env = env
        self.model_name = f"FFA__{os.path.split(path)[1].split('.')[0]}"

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

    def get_action(self, obs, *args):
        try:
            a = np.array([self.actions[self.counter]], dtype=float)
            a = np.clip(a, self.env.action_space.low[0], self.env.action_space.high[0])
        except IndexError as e:
            a = np.clip(np.array([0]), self.env.action_space.low[0], self.env.action_space.high[0])
            util.yellow(f"WARNING: Feedforward Controller at end of trajectory. Output will be 0 from now on.")
            self.trajectory_end = True
        self.counter += 1
        return np.array([a], dtype=float)

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
    "Q": np.diag([100, 100, 100, 100]),
    "R": 1,
}
F_LQR_2 = {
    "note": "LQR Q=np.diag([1000, 1000, 1000, 1000]) R=1",
    "F": np.array([[-31.622776601683427, -50.259303062636604, -246.50705065047896, -63.60946003330605]]),
    "eq": np.array([0, 0, 0, 0]),
    "Q": np.diag([1000, 1000, 1000, 1000]),
    "R": 1,
}
F_LQR_3 = {
    "note": "LQR Q=np.diag([1000, 1000, 0, 0]) R=1",
    "F": np.array([[-31.622776601684222, -23.899312329610922, -98.40512423898201, -21.037765755811748]]),
    "eq": np.array([0, 0, 0, 0]),
    "Q": np.diag([1000, 1000, 0, 0]),
    "R": 1,
}
F_LQR_4 = {
    "note": "LQR Q=np.diag([100, 100, 0, 0]) R=100",
    "F": np.array([[-0.9999999999998335, -1.8680009386439562, -26.925641920732488, -6.074588171203054]]),
    "eq": np.array([0, 0, 0, 0]),
    "Q": np.diag([100, 100, 0, 0]),
    "R": 1,
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
    "Q": np.diag([1000, 1000, 1000, 1000]),
    "R": 1,
}
F_NN_lin_1 = {
    "note": "NN cartpole_model__CartPoleContinous2Env___2023_03_02__16_40_49: linearized",
    "F": -np.array([[16.947, 31.633, 150.457, 40.509]]),
    "eq": np.array([0, 0, 0, 0]),
}
F_LQR_RWP_1 = {
    "note": "LQR Q = np.diag([30, 30, 30, 30]) R = 1",
    # "F": np.array([[-16550.973140478134, -1875.682378590107, -5.477225578913931, -6.718700759811327]]),
    "F": np.array([[-16550.973140478134, -1875.682378590107, 0, -6.718700759811327]]),
    "eq": np.array([0, 0, 0, 0]),
    "Q": np.diag([30, 30, 30, 30]),
    "R": 1
}
F_LQR_RWP_2 = {
    "note": "LQR  for 3 state, 0 added, Q = np.diag([30, 30, 30]) R = 1",
    "F": np.array([[-1.48662192e+04, -1.68475490e+03, 0, -5.47722557e+00]]),
    "eq": np.array([0, 0, 0]),
    "Q": np.diag([30, 30, 30]),
    "R": 1
}
F_PP_RWP_1 = {
    "note": "Pole Placement -4+-2j, -8 + nachträglich F[-1]*1e-2 verkleindert",
    "F": np.array( [[-7.93869720e-01, -8.82735934e-02, 0,  -6.53288091e-07]]),
    "eq": np.array([0, 0, 0]),
}
F_PP_RWP_2 = {
    "note": "??",
    "F": np.array( [[-356.215, -35.857, 0,  -0.043797]]),
    "eq": np.array([0, 0, 0]),
}
F_PP_BB_1 = {
    "note": "Pole Placement [-1.3+3.4j, -1.3-3.4j, -1.5+1.2j, -1.5-1.2j]",
    "F": np.array( [[-0.6300644137614678, -0.14085322764525993, 0.49484948, 0.1120112]]),
    "eq": np.array([0, 0, 0, 0]),
}
F_LQR_BB_1 = {
    "note": "LQR [-1.3+3.4j, -1.3-3.4j, -1.5+1.2j, -1.5-1.2j]",
    "F": np.array( [[-5.989644501647571, -8.434361610122082, 26.20980739989586, 5.572117832137562]]),
    "eq": np.array([0, 0, 0, 0]),
}