import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback
import pickle


class CustomCallback(BaseCallback):
    def __init__(self, log_path=None, verbose: int = 0):
        self.log_path = log_path
        super().__init__(verbose)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)
        self.parameter_file = os.path.join(self.log_path, "parameter_log.pickle")
        with open(self.parameter_file, "wb") as f:
            pickle.dump([], f)

    def _on_step(self) -> bool:
        return super()._on_step()

    def on_rollout_start(self) -> None:
        p = self.model.get_parameters()
        with open(self.parameter_file, "rb") as f:
            p_log = pickle.load(f)
        p_log.append(p)
        with open(self.parameter_file, "wb") as f:
            pickle.dump(p_log, f)
