import numpy as np
from collections import deque
import util


class RolyPolyAgent:
    def __init__(self, env, lower_eq_controller, upper_eq_controller, swingup_controller):
        self.env = env
        self.lower_eq_controller = lower_eq_controller
        self.upper_eq_controller = upper_eq_controller
        self.swingup_controller = swingup_controller
        self.controllers = [lower_eq_controller, swingup_controller, upper_eq_controller]

        # agent has 3 goals: 0: stabilize lower EQ, 1: swing up to near top, 2: stabilize upper EQ
        self.goal = 0

    def play(self):
        obs, _ = self.env.reset()
        self.last_states = deque([obs], maxlen=10)
        while True:
            # select controller
            current_agent = self.controllers[self.goal]

            # progress environment
            action = current_agent.get_action(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs = util.project_to_interval(obs)
            self.last_states.append(obs)

            # update state machine
            # pole has been at lower eqillibrium for some steps -> start swingup
            if (
                len(self.last_states) >= 10
                and np.abs(np.sum(np.abs(self.last_states)) - 10 * np.pi) < 0.1
                and self.goal == 0
            ):
                self.goal = 1
                print("Swingup")
            # pole is near upper EQ point -> start upper EQ controller
            elif abs(obs[2]) < 0.1 and self.goal == 1:
                if hasattr(self.swingup_controller, "reset_trajectory"):
                    self.swingup_controller.reset_trajectory()
                self.goal = 2
                print("Upper EQ")
            # pole left upper EQ -> go back to start
            elif self.goal == 2 and abs(obs[2]) > 0.4:
                self.goal = 0
                print("Lower EQ")

            if terminated or truncated:
                obs, _ = self.env.reset()
                self.goal = 0
                print("Lower EQ")
