import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import csv
from ipydex import IPS, activate_ips_on_exception

from envs.cartpole import (
    CartPoleDiscreteEnv,
    CartPoleContinous2Env,
    CartPoleContinousSwingupEnv,
    CartPoleContinous5StateEnv,
)
from envs.cartpole_transition import (
    CartPoleTransitionDiscreteEnv,
    CartPoleTransitionContinousEnv,
    CartPoleTransitionContinous2Env,
)
from envs.pendulum import StdPendulumEnv
from envs.reaction_wheel import DefaultReactionWheelEnv
from envs.ballbeam import DefaultBallBeamEnv
from envs.manipulator import ManipulatorEnv
from ppo.ppo_agent import PPOAgent
from manual.roly_poly import RolyPolyAgent
from classical.classical_control import *
import util
from callbacks.callback import *
import torch as th

activate_ips_on_exception()

np.random.seed(1)
folder_path = os.path.abspath(os.path.dirname(__file__))

### --- Mode --- ###
# mode = "train"
# mode = "retrain"
# mode = "play"
# mode = "eval"
# mode = "cooperative"
# mode = "manual"
# mode = "rwp_up_hold"
# mode = "pid"
# mode = "state_feedback"
# mode = "generate swingup trajectory"
# mode = "input from file"
# mode = "rp"
# mode = "compare"
mode = "test"


### --- Environment --- ###
# env = CartPoleContinousSwingupEnv()
# env = CartPoleContinous5StateEnv()
env = CartPoleContinous2Env()
# env = CartPoleDiscreteEnv()
env2 = CartPoleContinous2Env()
# env = CartPoleTransitionDiscreteEnv()
# env = CartPoleTransitionContinous2Env()

# env = StdPendulumEnv()
# env = DefaultReactionWheelEnv()
# env = DefaultBallBeamEnv()
env = ManipulatorEnv()

### --- Agent --- ###
# agent = PPOAgent(env, policy_kwargs={"net_arch": {'pi': [2000], 'vf': [100, 100]}})
# agent.action_net_kwargs = {
#     "bias": None,
#     "weight": -th.tensor([[-25.62277, -45.25930, -240.50705, -70.60946]], requires_grad=True)
#     }
# agent = PPOAgent(env)
# agent = JacobianApproximationControl(env)
# agent = ExactLinearizationAgent(env)
# agent = FeedbackAgent(env, F_LQR_BB_1)
# agent = ManualAgent(env)

### --- Callback --- ###
callback = CustomCallback()

if mode == "train":
    # env.render_mode = "human"
    print("Training")
    agent.train(total_timesteps=1e5, callback=callback, eval=False)  # , save_model=False)

if mode == "retrain":
    # env.render_mode = "human"
    model_name = "CartPoleContinous2Env___2023_04_12__10_36_25"
    assert agent.env.name in model_name, "wrong environment"
    agent.load_model(model_name)
    # agent.model.env = env
    print("continue Training")
    agent.train(1000000, callback=callback)

elif mode == "eval":
    print("Eval")
    # env.render_mode = "human"
    agent.load_model("CartPoleContinous2Env___2023_04_12__14_57_36")
    agent.eval()

elif mode == "play":
    print("Play")
    env.render_mode = "human"
    agent.load_model("DefaultBallBeamEnv___2023_09_13__15_25_27__good")
    agent.play(10)

elif mode == "cooperative":
    env.render_mode = "human"

    system = "cartpole"
    # system = "pendulum"

    if system == "cartpole":
        swingup_agent = (
            "CartPoleContinousSwingupEnv___2023_03_08__11_14_01"  # CartPoleContinousSwingupEnv___2023_03_08__11_16_25
        )
        balance_agent = "CartPoleContinous2Env___2023_03_06__15_13_42"
        catch_threshold = 0.5
    elif system == "pendulum":
        swingup_agent = "StdPendulumEnv___2023_07_11__14_09_12_swingup"
        balance_agent = "StdPendulumEnv___2023_07_11__14_52_05_upper"
        catch_threshold = np.pi / 180 * 45

    agent.load_model(swingup_agent)

    current_agent = swingup_agent
    obs, _ = env.reset()
    while True:
        if system == "cartpole":
            angle = obs[2]
        elif system == "pendulum":
            angle = env.state[0]
        if current_agent == swingup_agent and np.abs(util.project_to_interval(angle)) <= catch_threshold:
            current_agent = balance_agent
            agent.load_model(balance_agent)
            print("switching to balance")
        if current_agent == balance_agent and np.abs(util.project_to_interval(angle)) > catch_threshold:
            current_agent = swingup_agent
            agent.load_model(swingup_agent)
            print("switching to swingup")

        action, _states = agent.model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

elif mode == "manual":
    env.render_mode = "human"
    state, _ = env.reset(state=np.array([0.0, 0, 0, 0]))
    a = 0
    r = True
    t = 0
    while True:
        # a = np.array([(np.random.random(1)-0.5)*1e-3], dtype=np.float32)[0,0]
        # a = env.action_space.sample()
        # if env.ep_step_count > 100 and abs(state1[-1]) > 0 and r==True:
        #     a = - state[-1] * 0.0001
        a = agent.get_action(state, t)

        a = np.clip(a, env.action_space.low[0], env.action_space.high[0])
        if isinstance(env.action_space, gym.spaces.Box) and not isinstance(a, np.ndarray):
            a = np.array([a])
        state, reward, terminated, truncated, info = env.step(a)
        t += env.tau
        if terminated or truncated:
            state, _ = env.reset(state=np.array([0.1, 0, 0, 0]))
            a = 0

elif mode == "rwp_up_hold":
    env.render_mode = "human"
    state, _ = env.reset(state=np.array([np.pi-0.4, 0, 0, 0]))
    a = 0
    r = True
    s = 0
    controller = "up"
    integral = 0
    prevError = 0
    timeDelta = env.tau
    Kp = 2
    Ki = 0
    Kd = 0.1
    max_v = 500
    while True:
        if controller == "up":
            # swingup
            # speed up
            if s == 0:
                if state[-1] < max_v:
                    a = 0.001
                else:
                    s = 1
                    a = 0
            # detect turning point
            elif s == 1:
                if np.abs(state[1]) < 0.1 and state[0] < np.pi and state[-1] > 0:
                    s = 2
                if np.abs(state[1]) < 0.1 and state[0] > np.pi and state[-1] < 0:
                    s = 3
            elif s == 2:
                if state[-1] > 0:
                    a = -0.1
                elif state[-1] > -max_v:
                    a = -0.001
                else:
                    s = 1
                    a = 0
            elif s == 3:
                if state[-1] < 0:
                    a = 0.1
                elif state[-1] < max_v:
                    a = 0.001
                else:
                    s = 1
                    a = 0
            elif np.abs(util.project_to_interval(state[0])) < 0.1:
                controller = "PID"
                integral = 0
                prev_error = 0
        if controller == "PID":
            error = state[0]
            integral += error * timeDelta
            derivative = (error - prevError) / timeDelta
            prevError = error
            a = Kp * error + Ki * integral + Kd * derivative
            if np.abs(util.project_to_interval(state[0])) > 0.3:
                controller = "up"

        if isinstance(env.action_space, gym.spaces.Box) and not isinstance(a, np.ndarray):
            a = np.array([a])
        if controller == "PID" and not env.action_space.contains(a):
            integral -= error * timeDelta
        a = np.clip(a, env.action_space.low[0], env.action_space.high[0])
        state, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            state, _ = env.reset(state=np.array([np.pi, 0, 0, 0]))
            a = 0
            s = 0
            integral = 0
            prev_error = 0

elif mode == "pid":
    env.render_mode = "human"
    state, _ = env.reset(state=np.array([0.1, 0, 0, 0]))
    a = 0
    integral = 0
    prevError = 0
    timeDelta = env.tau
    Kp = 2
    Ki = 0
    Kd = 0.1
    while True:
        error = state[0]
        integral += error * timeDelta
        derivative = (error - prevError) / timeDelta
        prevError = error
        a = Kp * error + Ki * integral + Kd * derivative
        if isinstance(env.action_space, gym.spaces.Box) and not isinstance(a, np.ndarray):
            a = np.array([a])
        # arw
        if not env.action_space.contains(a):
            integral -= error * timeDelta

        a = np.clip(a, env.action_space.low[0], env.action_space.high[0])
        state, reward, terminated, truncated, info = env.step(a)
        if terminated or truncated:
            state, _ = env.reset(state=np.array([0.1, 0, 0, 0]))
            integral = 0
            prev_error = 0

elif mode == "state_feedback":
    env.render_mode = "human"
    state1, _ = env.reset(state=np.array([0.1, 0, 0, 0]))
    while True:
        F = F_PP_BB_1
        # state1 = util.project_to_interval(state1 - F["eq"], min=-np.pi, max=np.pi)
        u = -F["F"] @ np.array(state1)
        action = np.clip(u, env.action_space.low[0], env.action_space.high[0])
        state1, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            state1, _ = env.reset(state=np.array([0.1, 0, 0, 0]))

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
    agent1.load_model("CartPoleContinous2Env___2023_04_04__11_25_46__best")
    # F = F_LQR_2["F"]
    # agent1 = FeedbackAgent(env, F)

    ### --- Agents 2 --- ###
    agent2 = PPOAgent(env2)
    agent2.load_model("CartPoleContinous2Env___2023_04_04__11_10_59__x_off")
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

    env.render_mode = "human"
    state, _ = env.reset(state=np.array([0, 0, 0, 0, 0, 0]))
    diff = 10
    dec = 50
    counter = 0
    back_transition = False
    while True:
        action = 0
        if env.ep_step_count == 0:
            start_angle = state[1]
        if env.ep_step_count < diff:
            action = 10
        if (np.abs((state[1] - start_angle) % (np.pi*2)) < 0.07 \
            or np.abs((state[1] - start_angle) % (np.pi*2) - 2*np.pi) < 0.07) \
            and env.ep_step_count > diff:
            back_transition = True

        if back_transition:
            action = -10
            counter += 1

        if counter == 10:
            back_transition = False
            counter = 0

        # action = np.sin(env.ep_step_count/10)
        state, r, ter, tru, _ = env.step(action)
        if ter or tru:
            counter = 0
            back_transition = False
            env.reset()

