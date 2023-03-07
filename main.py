import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from envs.cartpole import CartPoleDiscreteEnv, CartPoleContinous2Env, CartPoleContinousSwingupEnv
from envs.cartpole_transition import (
    CartPoleTransitionDiscreteEnv,
    CartPoleTransitionContinousEnv,
    CartPoleTransitionContinous2Env,
)
from CrossEntropyLearning.cartpoleAgent1_gymnasium import Agent
from manual.manual import ManualAgent
from ppo.ppo_agent import PPOAgent
from classical.feedback_agent import *

np.random.seed(1)
folder_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(folder_path, "CrossEntropyLearning", "cartpole_transition_crossentropy.h5")

### --- Mode --- ###
# mode = "train"
mode = "retrain"
# mode = "play"
# mode = "manual"
# mode = "state_feedback"
# mode = "compare"

### --- Environment --- ###
env = CartPoleContinousSwingupEnv()
# env = CartPoleContinous2Env()
env2 = CartPoleContinous2Env()
# env = CartPoleTransitionDiscreteEnv()
# env = CartPoleTransitionContinous2Env()

### --- Agent --- ###
# agent = Agent(env)
agent = PPOAgent(env)
# agent = ManualAgent(env)


if mode == "train":
    print("Training")
    agent.train()
if mode == "retrain":
    model_name = "CartPoleContinousSwingupEnv___2023_03_07__15_28_05"
    assert agent.env.name in model_name, "wrong environment"
    agent.load_model(model_name)
    # agent.model.env = env
    print("continue Training")
    agent.train(1000000)
elif mode == "play":
    print("Play")
    # env = gym.make("CartPole-v1", render_mode="human")
    env.render_mode = "human"
    # agent.model.load_weights(model_path)
    agent.load_model("CartPoleContinousSwingupEnv___2023_03_07__14_54_48")
    agent.play(10)
elif mode == "manual":
    env.render_mode = "human"
    env.reset()
    a = 0
    while True:
        if isinstance(env.action_space, gym.spaces.Box) and not isinstance(a, np.ndarray):
            a = np.array([a])
        state1, reward, terminated, truncated, info = env.step(a)
        if terminated:
            obs, _ = env.reset()

elif mode == "state_feedback":
    env.render_mode = "human"
    state1, _ = env.reset()
    while True:
        F = F_LQR_3["F"]
        u = -F @ np.array(state1)
        action = np.clip(u, env.action_space.low[0], env.action_space.high[0])
        state1, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state1, _ = env.reset()

elif mode == "compare":
    ### --- Paras --- ###
    render = False

    ### --- Agents 1 --- ###
    agent1 = PPOAgent(env)
    agent1.load_model("CartPoleContinous2Env___2023_03_06__15_13_56")
    # F = F_LQR_2
    # agent1 = FeedbackAgent(env, F)

    ### --- Agents 2 --- ###
    agent2 = PPOAgent(env2)
    agent2.load_model("CartPoleContinous2Env___2023_03_07__15_35_12")
    # F = F_LQR_3
    # agent2 = FeedbackAgent(env2, F)

    ### ---------------- ###

    if render:
        env.render_mode = "human"
        env2.render_mode = "human"
    actions1 = []
    actions2 = []
    states1 = []
    states2 = []
    reset_freq = 500

    state1, _ = env.reset(seed=0)
    state2, _ = env2.reset(seed=0)
    for i in range(5000):
        states1.append(state1)
        states2.append(state2)

        action1 = agent1.get_action(state1)
        actions1.append(action1)

        action2 = agent2.get_action(state2)
        actions2.append(action2)

        state1, reward, terminated, truncated, info = env.step(action1)
        state2, reward2, terminated2, truncated2, info2 = env2.step(action2)
        if terminated or terminated2:
            state1, _ = env.reset(seed=i + 1)
            state2, _ = env.reset(seed=i + 1)

        # reset if pendulum is stable -> more interesting stablization
        if i % reset_freq == reset_freq - 1:
            state1, _ = env.reset(seed=i)
            state2, _ = env2.reset(seed=i)

    states1 = np.array(states1)
    states2 = np.array(states2)

    fig, ax = plt.subplots(2, 1)
    fig.suptitle(
        f"Comparison \n1 {agent1.__class__.__name__} {agent1.model_name}\n2 {agent2.__class__.__name__} {agent2.model_name}"
    )
    ax[0].plot(np.arange(len(states1)), states1[:, 0], label=f"{agent1.__class__.__name__} x", color="tab:blue")
    ax[0].plot(
        np.arange(len(states2)),
        states2[:, 0],
        label=f"{agent2.__class__.__name__} x",
        color="tab:cyan",
        linestyle="dashed",
    )
    ax[0].plot(np.arange(len(states1)), states1[:, 2], label=f"{agent1.__class__.__name__} phi", color="tab:red")
    ax[0].plot(
        np.arange(len(states2)),
        states2[:, 2],
        label=f"{agent2.__class__.__name__} phi",
        color="tab:pink",
        linestyle="dashed",
    )
    ax[0].set_title("States")
    ax[0].legend()
    ax[0].grid()

    labels = [f"1 {agent1.__class__.__name__}", f"2 {agent2.__class__.__name__}"]
    ax[1].plot(np.arange(len(actions1)), actions1, label=labels[0])
    ax[1].plot(np.arange(len(actions2)), actions2, label=labels[1], linestyle="dashed")
    ax[1].set_title("Actions")
    ax[1].legend()
    ax[1].grid()

    fig2, ax2 = plt.subplots(2, 1)
    ax2[0].plot(np.arange(len(states1)), states1[:, 0] - states2[:, 0], label=f"A1-A2 delta x", color="tab:blue")
    ax2[0].plot(np.arange(len(states1)), states1[:, 2] - states2[:, 2], label=f"A1-A2 delta phi", color="tab:red")
    ax2[0].set_title("State delta")
    ax2[0].legend()
    ax2[0].grid()

    ax2[1].plot(np.arange(len(actions1)), np.array(actions1) - np.array(actions2), label="A1-A2 actions")
    ax2[1].set_title("Action delta")
    ax2[1].legend()
    ax2[1].grid()
    plt.show()
