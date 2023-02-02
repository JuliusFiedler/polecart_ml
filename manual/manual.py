from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras import activations

class ManualAgent():
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

    def play(self, num_episodes, render=True):
        """Test the trained agent
        """
        state, _ = self.env.reset()
        while True:
            if render:
                self.env.render()
            action = 0
            state, reward, done, _, _ = self.env.step(action)
            # if done:
            #     break