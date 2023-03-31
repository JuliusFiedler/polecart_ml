import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import csv
from ipydex import IPS, activate_ips_on_exception

from envs.cartpole import CartPoleDiscreteEnv, CartPoleContinous2Env, CartPoleContinousSwingupEnv
from envs.cartpole_transition import (
    CartPoleTransitionDiscreteEnv,
    CartPoleTransitionContinousEnv,
    CartPoleTransitionContinous2Env,
)
from ppo.ppo_agent import PPOAgent
from manual.roly_poly import RolyPolyAgent
from classical.classical_control import *
import util
from callbacks.callback import *
import torch as th

activate_ips_on_exception()

np.random.seed(1)
folder_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(folder_path, "CrossEntropyLearning", "cartpole_transition_crossentropy.h5")

### --- Mode --- ###
mode = "train"
# mode = "retrain"
# mode = "play"
# mode = "eval"
# mode = "cooperative"
# mode = "manual"
# mode = "state_feedback"
# mode = "generate swingup trajectory"
# mode = "input from file"
# mode = "rp"
# mode = "compare"
# mode = "test"


### --- Environment --- ###
# env = CartPoleContinousSwingupEnv()
env = CartPoleContinous2Env()
env2 = CartPoleContinous2Env()
# env = CartPoleTransitionDiscreteEnv()
# env = CartPoleTransitionContinous2Env()

### --- Agent --- ###
# agent = PPOAgent(env, policy_kwargs={"net_arch": {'pi': [], 'vf': [64, 64]}})
# agent.action_net_kwargs = {
#     "bias": None,
#     "weight": -th.tensor([[-25.62277, -45.25930, -240.50705, -70.60946]], requires_grad=True)
#     }
agent = PPOAgent(env)
# agent = ManualAgent(env)

### --- Callback --- ###
callback = CustomCallback()

if mode == "train":
    # env.render_mode = "human"
    print("Training")
    agent.train(total_timesteps=3e5, callback=callback)

if mode == "retrain":
    # env.render_mode = "human"
    model_name = "CartPoleContinous2Env___2023_03_24__12_01_04"
    assert agent.env.name in model_name, "wrong environment"
    agent.load_model(model_name)
    # agent.model.env = env
    print("continue Training")
    agent.train(4000000)

elif mode == "eval":
    print("Eval")
    # env.render_mode = "human"
    agent.load_model("CartPoleContinous2Env___2023_03_16__11_03_00_good")
    agent.eval()

elif mode == "play":
    print("Play")
    env.render_mode = "human"
    agent.load_model("CartPoleContinous2Env___2023_03_28__17_31_21")
    agent.play(10)

elif mode == "cooperative":
    env.render_mode = "human"
    swingup_agent = (
        "CartPoleContinousSwingupEnv___2023_03_08__11_14_01"  # CartPoleContinousSwingupEnv___2023_03_08__11_16_25
    )
    balance_agent = "CartPoleContinous2Env___2023_03_06__15_13_42"
    agent.load_model(swingup_agent)

    catch_threshold = 0.5
    current_agent = swingup_agent
    obs, _ = env.reset()
    while True:
        if current_agent == swingup_agent and np.abs(util.project_to_interval(obs[2])) <= catch_threshold:
            current_agent = balance_agent
            agent.load_model(balance_agent)
            print("switching to balance")
        if current_agent == balance_agent and np.abs(util.project_to_interval(obs[2])) > catch_threshold:
            current_agent = swingup_agent
            agent.load_model(swingup_agent)
            print("switching to swingup")

        action, _states = agent.model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

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
    state1, _ = env.reset(state=np.array([0, 0, 0.1, 0]))
    while True:
        F = F_LQR_2
        state1 = util.project_to_interval(state1 - F["eq"], min=-np.pi, max=np.pi)
        u = -F["F"] @ np.array(state1)
        action = np.clip(u, env.action_space.low[0], env.action_space.high[0])
        state1, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state1, _ = env.reset()

elif mode == "generate swingup trajectory":
    env.render_mode = "human"
    agent = FeedbackAgent(env, F_LQR_LOWER_EQ_1)
    state1, _ = env.reset(state=np.array([0, 0, 0.1, 0]))
    actions = []
    states = []
    for i in range(400):
        states.append(state1)
        action = agent.get_action(state1)
        actions.append(action[0])
        state1, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            assert False, "there should be no reset during this process"
    actions.reverse()
    states.reverse()
    # get rid of slow start where nothing happens
    for i, action in enumerate(actions):
        if action >= 0.01:
            actions = actions[i:]
            states = states[i:]
            break
    with open("trajectories/cartpole_swingup_state_and_action.csv", mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        for i in range(len(actions)):
            writer.writerow([*states[i], actions[i]])

elif mode == "input from file":
    env.render_mode = "human"
    state1, _ = env.reset(state=np.array([0, 0, np.pi, 0]))
    with open("trajectories/cartpole_swingup_1.csv", newline="") as csvfile:
        actions = []
        reader = csv.reader(csvfile, delimiter=",")
        for row in reader:
            actions.append(row[0])
    for action in actions:
        state, _, _, _, _ = env.step(-np.array([action], dtype=float))
        if abs(state[2]) < 0.1:
            break
    print("swingup done")

    while True:
        F = F_LQR_2
        state1 = util.project_to_interval(state1 - F["eq"], min=-np.pi, max=np.pi)
        u = -F["F"] @ np.array(state1)
        action = np.clip(u, env.action_space.low[0], env.action_space.high[0])
        state1, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state1, _ = env.reset()

elif mode == "rp":
    env.render_mode = "human"
    # L_EQ_agent = FeedbackAgent(env, F_LQR_LOWER_EQ_1)
    # U_EQ_agent = FeedbackAgent(env, F_LQR_2)
    # swingup_agent = FeedforwardAgent(env, "cartpole_swingup_1.csv")

    L_EQ_agent = FeedbackAgent(env, F_LQR_LOWER_EQ_1)
    U_EQ_agent = PPOAgent(env)
    U_EQ_agent.load_model("CartPoleContinous2Env___2023_03_06__15_13_42")
    swingup_agent = PPOAgent(env)
    swingup_agent.load_model("CartPoleContinousSwingupEnv___2023_03_09__14_51_57")

    agent = RolyPolyAgent(env, L_EQ_agent, U_EQ_agent, swingup_agent)
    agent.play()

elif mode == "compare":
    ### --- Paras --- ###
    render = False

    ### --- Agents 1 --- ###
    agent1 = PPOAgent(env)
    agent1.load_model("CartPoleContinous2Env___2023_03_06__15_13_56")
    # F = F_LQR_2["F"]
    # agent1 = FeedbackAgent(env, F)

    ### --- Agents 2 --- ###
    agent2 = PPOAgent(env2)
    agent2.load_model("CartPoleContinous2Env___2023_03_07__15_35_12")
    # F = F_LQR_3["F"]
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

elif mode == "test":
    from ppo.supervised import PretrainedNet
    import torch

    net = PretrainedNet()
    net.load_state_dict(torch.load("trajectories/swingupNN.pth"))

    env.render_mode = "human"
    state, _ = env.reset(state=np.array([0, 0, np.pi, 0]))
    while True:
        action = net(torch.from_numpy(state)).detach().numpy()
        action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
        state, r, ter, tru, _ = env.step(action)
        if abs(state[2]) < 0.1:
            print("swingup done")
        if ter:
            state, _ = env.reset()
