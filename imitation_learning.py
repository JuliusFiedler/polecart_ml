import sys, os
import numpy as np
import gym
import gymnasium
import pickle
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper

from wrappers.wrappers import GymnasiumToGymWrapper, BCWrapper
from envs.cartpole import CartPoleContinous2Env, CartPoleContinousSwingupEnv
from ppo.ppo_agent import PPOAgent
from classical.classical_control import *

# sys.modules["gym"] = gymnasium

rng = np.random.default_rng(0)
# og_env = CartPoleContinous2Env()
og_env = CartPoleContinousSwingupEnv()
# og_env.render_mode = "human"
env = GymnasiumToGymWrapper(og_env)


# agent_type = "PPO"
agent_type = "state_feedback"


if agent_type == "PPO":
    agent = PPOAgent(env)
    name = "CartPoleContinous2Env___2023_03_16__11_03_00_good"
    agent.load_model(name)
    expert = agent.model
elif agent_type == "state_feedback":
    agent = FeedforwardAgent(env, "cartpole_swingup_1.csv")
    # agent = FeedbackAgent(env, F_LQR_2)
    expert = agent.get_action
    name = agent.model_name

folder_name = os.path.join("models", "Imitation__" + name)
os.makedirs(folder_name, exist_ok=True)


rollout_file_name = os.path.join(folder_name, "rollouts.pickle")
in1 = input("Use existing rollouts? (y/n)")
if os.path.exists(rollout_file_name) and in1 == "y":
    print("loading rollouts")
    with open(rollout_file_name, "rb") as f:
        rollouts = pickle.load(f)
else:
    print("collecting rollouts")
    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    with open(rollout_file_name, "wb") as f:
        pickle.dump(rollouts, f)

transitions = rollout.flatten_trajectories(rollouts)


bc_trainer = BCWrapper(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
    rng=rng,
)
print("train")
bc_trainer.train(n_epochs=30)
bc_trainer.save_policy(os.path.join(folder_name, "model.h5"))
reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
print("Reward:", reward)

env.env.render_mode = "human"
for i in range(10):
    obs = env.reset()
    while True:
        action, _ = bc_trainer.policy.predict(obs, deterministic=True)
        obs, rew, done, info = env.step(action)
        if done:
            break
