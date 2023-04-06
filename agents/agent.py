import torch as th
import os
import numpy as np
import matplotlib.pyplot as plt
from abc import abstractmethod


class BaseAgent:
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def get_action(self, obs):
        raise NotImplementedError()

    @abstractmethod
    def get_value(self, obs):
        raise NotImplementedError()

    @abstractmethod
    def get_real_state_value(self, obs):
        raise NotImplementedError

    def run_eval_episodes(self):
        xs = []
        phis = []
        inside_tol = []
        total_cost = 0
        episodes = 3
        steps = 1000
        for i in range(episodes):
            obs, _ = self.env.reset(state=np.array(self.env.c.EVAL_STATE) / (i + 2))
            for j in range(steps):
                xs.append(obs[0])
                phis.append(obs[2])
                action = self.get_action(obs)
                obs, reward, done, trunc, info = self.env.step(action)
                # add cost
                total_cost += (obs.T @ self.env.c.Q @ obs + action.T * self.env.c.R * action)[0]
                if done:
                    break

        phis.reverse()
        xs.reverse()
        phi_tolerance = self.env.c.phi_tolerance
        x_tolerance = self.env.c.x_tolerance
        for i in range(episodes):
            for j, phi in enumerate(phis[i * 1000 : (i + 1) * 1000]):
                if np.abs(phi) > phi_tolerance or np.abs(xs[steps * i + j]) > x_tolerance:
                    inside_tol.append((episodes - i) * steps - j)
                    break
        phis.reverse()
        xs.reverse()

        tol_lines = [-phi_tolerance, phi_tolerance, -x_tolerance, x_tolerance]
        plt.hlines(tol_lines, xmin=0, xmax=steps * episodes, colors="gray", linewidth=0.5)
        plt.plot(np.arange(len(xs)), xs, label=r"$x$")
        plt.plot(np.arange(len(phis)), phis, label=r"$\varphi$")
        plt.vlines(inside_tol, ymin=min(phis), ymax=max(phis), colors="red")
        plt.legend()
        plt.title("Evaluation Episodes")
        path = os.path.join("models", self.model_name, "eval.pdf")
        plt.savefig(path, format="pdf")
        plt.clf()

        with open(os.path.join("models", self.model_name, "eval.txt"), "w") as f:
            f.write(f"total cost: {round(total_cost, 4)}\n")
            f.write(f"av cost per step: {round(total_cost / (episodes*steps), 4)}\n")
            f.write(f"mean steps until steady state: {round(np.mean(inside_tol) - steps, 4)}\n")
