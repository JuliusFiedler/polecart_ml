import unittest
import numpy as np
from ipydex import IPS

from ppo.ppo_agent import PPOAgent
from envs.cartpole import CartPoleContinous2Env

"""
python -m unittest ut.test -v
"""


class TestStringMethods(unittest.TestCase):
    def test_ppo_seeding(self):
        env1 = CartPoleContinous2Env()
        env2 = CartPoleContinous2Env()

        agent_1 = PPOAgent(env1)
        agent_1.train(total_timesteps=50, save_model=False)

        agent_2 = PPOAgent(env2)
        agent_2.train(total_timesteps=50, save_model=False)

        self.assertEqual(agent_1.seed, agent_2.seed, msg="Seeds are unequal")

        action_net_1 = agent_1.model.policy.action_net.weight.detach().numpy()
        action_net_2 = agent_2.model.policy.action_net.weight.detach().numpy()

        self.assertEqual(np.sum(action_net_1 - action_net_2), 0, msg="Action net weights unequal")

        obs = np.array([1.1, 0.4, -0.3, 2])
        action_1 = agent_1.model.predict(obs, deterministic=True)[0]
        action_2 = agent_2.model.predict(obs, deterministic=True)[0]

        self.assertEqual(np.sum(action_1 - action_2), 0, msg="Predicted Actions unequal")


if __name__ == "__main__":
    unittest.main()
