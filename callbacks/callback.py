import numpy as np
import os
from stable_baselines3.common.callbacks import BaseCallback


class CustomCallback(BaseCallback):
    def __init__(self, log_path=None, verbose: int = 0):
        self.log_path = log_path
        super().__init__(verbose)

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.log_path is not None:
            os.makedirs(self.log_path, exist_ok=True)

    def _on_step(self) -> bool:
        return super()._on_step()
