import numpy as np
from ipydex import IPS

from ppo.cartpole_ppo import PPOAgent
from envs.cartpole_transition import CartPoleTransitionContinous2Env

env = CartPoleTransitionContinous2Env()

agent = PPOAgent(env)

agent.load_model("cartpole_model__CartPoleTransitionContinous2Env___2023_03_01__16_55_56.h5")
model = agent.model

input_features = 5
output_features = 1
policy_net_architecture = [64, 64]
action_net_architecture = [64]

policy_net_weights = []
policy_net_biases = []
action_net_weights = []
action_net_biases = []

for i in range(len(policy_net_architecture)):
    policy_net_weights.append(np.array(model.policy.state_dict()[f"mlp_extractor.policy_net.{2*i}.weight"]))
    policy_net_biases.append(np.array(model.policy.state_dict()[f"mlp_extractor.policy_net.{2*i}.bias"]))

if len(action_net_architecture) == 1:
    action_net_weights.append(np.array(model.policy.state_dict()[f"action_net.weight"]))
    action_net_biases.append(np.array(model.policy.state_dict()[f"action_net.bias"]))
else:
    raise NotImplementedError
    for i in range(len(action_net_architecture)):
        action_net_weights.append(np.array(model.policy.state_dict()[f"action_net.{2*i}.weight"]))
        action_net_biases.append(np.array(model.policy.state_dict()[f"action_net.{2*i}.bias"]))

activation_fn_class = model.policy.activation_fn

if activation_fn_class.__name__ == "Tanh":
    act_fn = np.tanh
else:
    raise NotImplementedError

def NN(obs):
    obs = np.array(obs)
    layer = obs
    for i in range(len(policy_net_architecture)):
        layer = policy_net_weights[i] @ layer + policy_net_biases[i]
        layer = act_fn(layer)
    
    for i in range(len(action_net_architecture)):
        layer = action_net_weights[i] @ layer + action_net_biases[i]
        layer = act_fn(layer)
    
    return layer

obs = [1,1,1,1,1]
NN(obs)
model.predict(obs, deterministic=True)

print("done")