import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from envs.cartpole import CartPoleDiscreteEnv, CartPoleContinous2Env
from envs.cartpole_transition import CartPoleTransitionDiscreteEnv, CartPoleTransitionContinousEnv, CartPoleTransitionContinous2Env
from CrossEntropyLearning.cartpoleAgent1_gymnasium import Agent
from manual.manual import ManualAgent
from ppo.cartpole_ppo import PPOAgent
from classical.feedback_agent import *

np.random.seed(1)
folder_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(folder_path, "CrossEntropyLearning", "cartpole_transition_crossentropy.h5")

### --- Mode --- ###
mode = "train"
# mode = "play"
# mode = "manual"
# mode = "state_feedback"
# mode = "compare"

### --- Environment --- ###
# env = CartPoleDiscreteEnv()
env = CartPoleContinous2Env()
env2 = CartPoleContinous2Env()
# env = CartPoleTransitionDiscreteEnv()
# env = CartPoleTransitionContinous2Env()

### --- Agent --- ###
# agent = Agent(env)
agent = PPOAgent(env)
# agent = ManualAgent(env)


if mode == "train":
    print("Training")
    # agent.train(percentile=70, num_iterations=15, num_episodes=100)
    agent.train()
    # agent.model.save(model_path, overwrite =True)
elif mode == "play":
    print("Play")
    # env = gym.make("CartPole-v1", render_mode="human")
    env.render_mode="human"
    # agent.model.load_weights(model_path)
    agent.load_model("cartpole_model__CartPoleContinous2Env___2023_03_02__15_05_25.h5")
    agent.play(10)
elif mode == "manual":
    env.render_mode="human"
    env.reset()
    a = 0
    while True:
        if isinstance(env.action_space, gym.spaces.Box) and not isinstance(a, np.ndarray):
            a = np.array([a])
        state1, reward, terminated, truncated, info = env.step(a)
        if terminated:
            obs, _ = env.reset()

elif mode == "state_feedback":
    env.render_mode="human"
    state1, _ = env.reset()
    while True:
        F = F_LQR_3
        u = - F @ np.array(state1)
        action = np.clip(u, env.action_space.low[0], env.action_space.high[0])
        print(u)
        state1, reward, terminated, truncated, info = env.step(action)
        if terminated:
            state1, _ = env.reset()

elif mode == "compare":
    ### --- Paras --- ###
    render = False
    
    ### --- Agents 1 --- ###
    # agent1 = PPOAgent(env)
    # agent1.load_model("cartpole_model__CartPoleContinous2Env___2023_03_02__15_05_25.h5")
    F = F_LQR_2
    agent1 = FeedbackAgent(env, F)
    
    
    ### --- Agents 2 --- ###
    F = F_LQR_1
    agent2 = FeedbackAgent(env2, F)
    
    ### ---------------- ###
    
    
    if render:
        env.render_mode = "human"
        env2.render_mode = "human"
    actions1 = []
    actions2 = []
    reset_freq = 500
    
    state1, _ = env.reset(seed=0)
    state2, _ = env2.reset(seed=0)
    for i in range(5000):
        action1 = agent1.get_action(state1)
        actions1.append(action1)
        
        action2 = agent2.get_action(state2)
        actions2.append(action2)
                
        state1, reward, terminated, truncated, info = env.step(action1)
        state2, reward2, terminated2, truncated2, info2 = env2.step(action2)
        if terminated or terminated2:
            state1, _ = env.reset(seed=i+1)
            state2, _ = env.reset(seed=i+1)
            
        # reset if pendulum is stable -> more interesting stablization
        if i % reset_freq == reset_freq - 1:
            state1, _ = env.reset(seed=i)
            state2, _ = env2.reset(seed=i)
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(np.arange(len(actions1)), actions1, label=f"1 {agent1.__class__.__name__}")
    ax[0].plot(np.arange(len(actions2)), actions2, label=f"2 {agent2.__class__.__name__}", linestyle="dashed")
    
    ax[1].plot(np.arange(len(actions1)), np.array(actions1) - np.array(actions2), label="1-2")
    plt.legend()
    plt.show()
    
    
