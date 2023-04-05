import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import pickle
import itertools as it
import torch as th
from ipydex import IPS, activate_ips_on_exception

from ppo.ppo_agent import PPOAgent
from envs.cartpole import CartPoleContinous2Env

activate_ips_on_exception()


name = "CartPoleContinous2Env___2023_04_04__11_25_46__best"

path = os.path.join("models", name)

env = CartPoleContinous2Env(render_mode=False)
agent = PPOAgent(env)

agent.load_model(name)


n_states = env.observation_space.shape[0]
if not n_states == 4:
    raise NotImplementedError()
labels = ["x", "xdot", "phi", "phidot"]
low = np.array([-2, -10, -np.pi / 2, -10])
high = -low
resolution = 1000
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
        actions = agent.model.policy.predict(state, deterministic=True)[0]
        values = agent.model.policy.predict_values(state)[:, 0].detach().numpy()
        action_grid[:, i] = actions[:, 0]
        value_grid[:, i] = values

    ax_action = fig_action.add_subplot(3, 2, idx + 1, projection="3d")
    ax_value = fig_value.add_subplot(3, 2, idx + 1, projection="3d")

    X, Y = np.meshgrid(X, Y)
    # Plot the surface.
    surf_action = ax_action.plot_surface(X, Y, action_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    surf_value = ax_value.plot_surface(X, Y, value_grid, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax_action.set(xlabel=labels[comb[0]], ylabel=labels[comb[1]], zlabel="action")
    ax_value.set(xlabel=labels[comb[0]], ylabel=labels[comb[1]], zlabel="value")

    # Add a color bar which maps values to colors.
    fig_action.colorbar(surf_action, shrink=0.5, aspect=5)
    fig_value.colorbar(surf_value, shrink=0.5, aspect=5)

fig_action.suptitle("Action Net")
fig_value.suptitle("Value Net")

fig_action.savefig(os.path.join(path, "action_net.pdf"), format="pdf")
fig_value.savefig(os.path.join(path, "value_net.pdf"), format="pdf")
