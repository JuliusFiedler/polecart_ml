import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
from ipydex import IPS, activate_ips_on_exception
activate_ips_on_exception()

from envs.cartpole import CartPoleEnv

from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import activations

class Agent:
    """Agent class for the cross-entrpoy learning algorithm
    """

    def __init__(self, env):
        """Set up the environment, the neural network and member variables

        Args:
            env ([gym.Environment]): [The game environment]

        """
        self.env = env
        self.observations = self.env.observation_space.shape[0]
        self.actions = self.env.action_space.n
        self.model = self.get_model()

    def get_model(self):
        """returns a Keras NN Model
        """
        model = Sequential()
        model.add(Dense(units=100, input_dim=self.observations)) # input: number of features
        model.add(Activation("relu"))
        model.add(Dense(units=self.actions)) # output: number of actions
        model.add(Activation("softmax"))
        model.summary()
        model.compile(
            loss="categorical_crossentropy",
            optimizer=Adam(learning_rate=0.001),
            metrics=["accuracy"]
        )
        return model

    def get_action(self, state):
        """Based o the state, get action
        """
        state = state.reshape(1, -1)
        action = self.model(state, training=False).numpy()[0]
        action = np.random.choice(self.actions, p=action)
        return action

    def get_samples(self, num_episodes):
        """Sample games
        """
        rewards = [0.0 for i in range(num_episodes)] # rewards of each episode
        episodes = [[] for i in range(num_episodes)] # sequence of states and actions per episode

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0.0
            while True:
                action = self.get_action(state)
                new_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                episodes[episode].append((state, action))
                state = new_state
                if done:
                    rewards[episode] = total_reward
                    break

        return rewards, episodes

    def filter_episodes(self, rewards, episodes, percentile):
        """helper function for training
        """
        reward_bound = np.percentile(rewards, percentile)
        x_train, y_train = [], []
        for reward, episode in zip(rewards, episodes):
            if reward >= reward_bound:
                observation = [step[0] for step in episode]
                action = [step[1] for step in episode]
                x_train.extend(observation)
                y_train.extend(action)
        x_train = np.asarray(x_train)
        y_train = to_categorical(y_train, num_classes=self.actions)
        return x_train, y_train, reward_bound

    def train(self, percentile, num_iterations, num_episodes):
        """Play games and train NN
        """
        for iteration in range(num_iterations):
            rewards, episodes = self.get_samples(num_episodes)
            x_train, y_train, reward_bound = self.filter_episodes(rewards, episodes, percentile)
            self.model.fit(
                x=x_train,
                y=y_train,
                verbose=0,
            )
            reward_mean = np.mean(rewards)
            print(f"Iteration: {iteration} Reward mean: {reward_mean}, reward bound: {reward_bound}")
            # if reward_mean > 300:
            #     break

    def play(self, num_episodes, render=True):
        """Test the trained agent
        """
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            total_reward = 0.0
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print(f"Total reward: {total_reward} in episode {episode + 1}")
                    break



