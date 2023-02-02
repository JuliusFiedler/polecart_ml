import os
from envs.cartpole import CartPoleEnv
from envs.cartpole_transition import CartPoleTransitionEnv
from CrossEntropyLearning.cartpoleAgent1_gymnasium import Agent
from manual.manual import ManualAgent


folder_path = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(folder_path, "CrossEntropyLearning", "cartpole_transition_crossentropy.h5")

mode = "manual"
# env = CartPoleEnv()
env = CartPoleTransitionEnv()
agent = Agent(env)
# agent = ManualAgent(env)
if mode == "train":
    print(agent.observations)
    print(agent.actions)
    print("Training")
    agent.train(percentile=70, num_iterations=15, num_episodes=100)
    agent.model.save(model_path, overwrite =True)
elif mode == "play":
    print("Play")
    # env = gym.make("CartPole-v1", render_mode="human")
    env.render_mode="human"
    agent.model.load_weights(model_path)
    agent.play(num_episodes=10)
elif mode == "manual":
    env.render_mode="human"
    env.reset()
    while True:
        action = 0
        state, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            env.reset()
