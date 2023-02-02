import os
from envs.cartpole import CartPoleEnv
from envs.cartpole_transition import CartPoleTransitionEnv
from CrossEntropyLearning.cartpoleAgent1_gymnasium import Agent
from manual.manual import ManualAgent
from ppo.cartpole_ppo import PPOAgent


folder_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(folder_path, "CrossEntropyLearning", "cartpole_transition_crossentropy.h5")

### --- Mode --- ###
# mode = "train"
mode = "play"
# mode = "manual"

### --- Environment --- ###
# env = CartPoleEnv()
env = CartPoleTransitionEnv()

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
        env.render()
        if terminated:
            env.reset()
