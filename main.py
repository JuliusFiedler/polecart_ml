import os
import numpy as np

from envs.cartpole import CartPoleEnv
from envs.cartpole_transition import CartPoleTransitionDiscreteEnv, CartPoleTransitionContinousEnv, CartPoleTransitionContinous2Env
from CrossEntropyLearning.cartpoleAgent1_gymnasium import Agent
from manual.manual import ManualAgent
from ppo.cartpole_ppo import PPOAgent
from classical.state_feedback import F


folder_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(folder_path, "CrossEntropyLearning", "cartpole_transition_crossentropy.h5")

### --- Mode --- ###
# mode = "train"
mode = "play"
# mode = "manual"
# mode = "state_feedback"

### --- Environment --- ###
# env = CartPoleEnv()
env = CartPoleTransitionDiscreteEnv()
# env = CartPoleTransitionContinousEnv()

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
    agent.load_model("ppo_cartpole_model_2023_02_02__15_28_08.h5")
    agent.play(10)
elif mode == "manual":
    env.render_mode="human"
    env.reset()
    a = 0
    while True:
        state, reward, terminated, truncated, info = env.step(a)
        if terminated:
            obs, _ = env.reset()

elif mode == "state_feedback":
    env.render_mode="human"
    state, _ = env.reset()
    while True:
        F2 = np.array([[ 654.79653851, -399.5922528,   279.5507985,  -253.5507985 ]])
        u = - F2 @ np.array(state[0:-1])
        action = np.clip(u, env.action_space.low[0], env.action_space.high[0])
        print(u)
        # if u > 0:
        #     action = 0
        # else:
        #     action = 1
        state, reward, terminated, truncated, info = env.step(action)
        if terminated:
            state, _ = env.reset()