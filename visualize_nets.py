import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
import pickle
import itertools as it
import torch as th
from ipydex import IPS, activate_ips_on_exception

from ppo.ppo_agent import PPOAgent
from envs.cartpole import CartPoleContinous2Env
from classical.classical_control import FeedbackAgent, F_LQR_2

activate_ips_on_exception()


def visualize_actions_values(env, agent, path, name, show_plot=False):
    n_states = env.observation_space.shape[0]
    if not n_states == 4:
        raise NotImplementedError()
    labels = ["x", "xdot", "phi", "phidot"]
    low = np.array([-1, -2, -np.pi / 180 * 20, -2])
    high = -low
    resolution = 1001
    fig_action = plt.figure(figsize=(10, 15))
    fig_value = plt.figure(figsize=(10, 15))

    for idx, comb in enumerate(it.combinations(np.arange(n_states), 2)):
        X = np.linspace(low[comb[0]], high[comb[0]], resolution)
        Y = np.linspace(low[comb[1]], high[comb[1]], resolution)

        action_grid = np.zeros((resolution, resolution))
        value_grid = np.zeros((resolution, resolution))
        for i, y in enumerate(Y):
            state = np.zeros((resolution, n_states))
            state[:, comb[0]] = X
            state[:, comb[1]] = y * np.ones(resolution)
            state = th.from_numpy(state)
            actions = agent.get_action(state)
            values = agent.get_value(state)
            assert action_grid[:, i].shape == actions[:, 0].shape
            action_grid[:, i] = actions[:, 0]
            assert value_grid[:, i].shape == values.shape
            value_grid[:, i] = values

        ax_action = fig_action.add_subplot(3, 2, idx + 1, projection="3d")
        ax_value = fig_value.add_subplot(3, 2, idx + 1, projection="3d")

        X, Y = np.meshgrid(X, Y)
        # Plot the surface.
        surf_action = ax_action.plot_surface(X, Y, action_grid, cmap="coolwarm", linewidth=0, antialiased=False)
        surf_value = ax_value.plot_surface(X, Y, value_grid, cmap="coolwarm", linewidth=0, antialiased=False)

        # wall projections
        # ax_action.contourf(X, Y, action_grid, zdir='z', offset=np.min(X), cmap='coolwarm')
        # ax_action.contourf(X, Y, action_grid, zdir='x', offset=np.min(Y), cmap='coolwarm')
        # ax_action.contourf(X, Y, action_grid, zdir='y', offset=np.min(action_grid), cmap='coolwarm')

        # ax_value.contourf(X, Y, value_grid, zdir='z', offset=np.min(X), cmap='coolwarm')
        # ax_value.contourf(X, Y, value_grid, zdir='x', offset=np.min(Y), cmap='coolwarm')
        # ax_value.contourf(X, Y, value_grid, zdir='y', offset=np.min(value_grid), cmap='coolwarm')

        ax_action.set(xlabel=labels[comb[0]], ylabel=labels[comb[1]], zlabel="action")  # , xlim=(np.min(X), np.max(X)),
        #   ylim=(np.min(Y), np.max(Y)), zlim=(np.min(action_grid), np.max(action_grid)))
        ax_value.set(xlabel=labels[comb[0]], ylabel=labels[comb[1]], zlabel="value")  # , xlim=(np.min(X), np.max(X)),
        #   ylim=(np.min(Y), np.max(Y)), zlim=(np.min(value_grid), np.max(value_grid)))

        # Add a color bar which maps values to colors.
        fig_action.colorbar(surf_action, shrink=0.5, aspect=5)
        fig_value.colorbar(surf_value, shrink=0.5, aspect=5)

    fig_action.suptitle(f"{name}\nAction Net")
    fig_value.suptitle(f"{name}\nValue Net")
    os.makedirs(path, exist_ok=True)

    fig_action.savefig(os.path.join(path, "action_net.pdf"), format="pdf")
    fig_value.savefig(os.path.join(path, "value_net.pdf"), format="pdf")
    if show_plot:
        plt.show()


if __name__ == "__main__":
    env = CartPoleContinous2Env(render_mode=False)

    # name = "CartPoleContinous2Env___2023_04_04__11_25_46__best"
    # agent = PPOAgent(env)
    # agent.load_model(name)

    agent = FeedbackAgent(env, F_LQR_2)
    name = agent.model_name

    path = os.path.join("models", name)

    visualize_actions_values(env, agent, path, name)
