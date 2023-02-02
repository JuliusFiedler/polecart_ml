import gym
import numpy as np
from ipydex import IPS

class Agent:
    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        if state[2] >= 0 and state[3] >= 0: # /
            action = 1 # right
        elif state[2] < 0 and state[3] < 0: # \
            action = 0 # left
        else:
            action = self.env.action_space.sample() # None
        return action

    def play(self, episodes, render=True):
        rewards = [0 for i in range(episodes)]

        for episode in range(episodes):
            state, _ = self.env.reset()
            total_reward = 0.0

            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                if done:
                    rewards[episode] = total_reward
                    break
        return rewards

if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode="human")
    agent = Agent(env)
    rewards = agent.play(episodes=5, render=True)

    rewards_mean = np.mean(rewards)
    rewards_min = np.min(rewards)
    rewards_max = np.max(rewards)
    print("mean: ", rewards_mean)
    print("min ", rewards_min)
    print("max ", rewards_max)