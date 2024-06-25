import gym
import os
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import seals  # needed to load environments
from ipydex import IPS, activate_ips_on_exception
activate_ips_on_exception()


env = gym.make("seals/CartPole-v0")
expert = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
)
# expert.learn(1000)  # Note: set to 100000 to train a proficient expert
path = os.path.join("models", "CartPoleDiscreteEnv___2023_05_11__16_29_35", "model.h5")
expert.set_parameters(path)

from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    make_vec_env(
        "seals/CartPole-v0",
        n_envs=5,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        rng=rng,
    ),
    rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    rng=rng,
)

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

import gym
import seals


venv = make_vec_env("seals/CartPole-v0", n_envs=8, rng=rng)
learner = PPO(
    env=venv,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
)
reward_net = BasicShapedRewardNet(
    venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
)
airl_trainer = AIRL(
    demonstrations=rollouts,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=2048,
    n_disc_updates_per_round=4,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
)

learner_rewards_before_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)
airl_trainer.train(20000)  # Note: set to 300000 for better results
learner_rewards_after_training, _ = evaluate_policy(
    learner, venv, 100, return_episode_rewards=True
)
import datetime as dt
t = dt.datetime.now().strftime("_%Y_%m_%d__%H_%M_%S")
folder_path = os.path.join("models", "IRL" + t)
os.makedirs(folder_path)
model_path = os.path.join(folder_path, "model.h5")
learner.save(model_path)
print(f"Model saved at {model_path}")

import matplotlib.pyplot as plt
import numpy as np

print("mean reward before:", np.mean(learner_rewards_before_training))
print("mean reward after:", np.mean(learner_rewards_after_training))

# plt.hist(
#     [learner_rewards_before_training, learner_rewards_after_training],
#     label=["untrained", "trained"],
# )
# plt.legend()
# plt.show()

obs = env.reset()
while True:
    obs, rew, term, info = env.step(learner.predict(obs, deterministic=True)[0])
    env.render("human")
    if term:
        obs = env.reset()