import numpy as np

class FeedbackAgent():
    def __init__(self, env, F) -> None:
        self.env = env
        self.F = F
        self.model_name = F["note"]
    
    def get_action(self, state):
        u = - self.F["F"] @ np.array(state)
        action = np.clip(u, self.env.action_space.low[0], self.env.action_space.high[0])
        return action
    
F_EV_imag_1 = {
    "note": "place eigenvalues: [-1.5+a*1j, -1.5-a*1j, -1.3 + b*1j, -1.3 - b*1j]",
    "F": np.array([[-2.4919724770642167, -2.514984709480122, -23.425986238532126, -4.057492354740066]]) 
}
F_EV_real_1 = {
    "note": "place eigenvalues: [-8., -7., -6., -5.]",
    "F": np.array([[-85.62691131487867, -54.33231396529075, -178.12345565736146, -40.166156982641404]]) 
}
F_LQR_1 = {
    "note": "LQR Q=np.diag([100, 100, 100, 100]) R=1",
    "F": np.array([[-9.999999999999986, -16.266056714636235, -90.5387468119659, -22.64298172774513]])
}
F_LQR_2 = {
    "note": "LQR Q=np.diag([1000, 1000, 1000, 1000]) R=1",
    "F": np.array([[-31.622776601683427, -50.259303062636604, -246.50705065047896, -63.60946003330605]])
}
F_LQR_3 = {
    "note": "LQR Q=np.diag([1000, 1000, 0, 0]) R=1",
    "F": np.array([[-31.622776601684222, -23.899312329610922, -98.40512423898201, -21.037765755811748]])
}