import numpy as np
import torch as th
from ipydex import IPS


from ppo.cartpole_ppo import PPOAgent
from envs.cartpole_transition import CartPoleTransitionContinous2Env

env = CartPoleTransitionContinous2Env()

agent = PPOAgent(env)

agent.load_model("cartpole_model__CartPoleTransitionContinous2Env___2023_03_01__16_55_56.h5")
model = agent.model

input_features = 5
output_features = 1

policy_net_weights = []
policy_net_biases = []
policy_net_activation_fns = []
action_net_weights = []
action_net_biases = []
action_net_activation_fns = []

# extract policy net information
if isinstance(model.policy.mlp_extractor.policy_net, th.nn.modules.container.Sequential):
    for item in model.policy.mlp_extractor.policy_net:
        if isinstance(item, th.nn.modules.linear.Linear):
            policy_net_weights.append(item.weight.detach().numpy())
            policy_net_biases.append(item.bias.detach().numpy())
        elif isinstance(item, th.nn.modules.activation.Tanh):
            policy_net_activation_fns.append(np.tanh)
        else:
            raise NotImplementedError
else:
    raise NotImplementedError
# extract action net information
if isinstance(model.policy.action_net, th.nn.modules.container.Sequential):
    for item in model.policy.action_net:
        if isinstance(item, th.nn.modules.linear.Linear):
            action_net_weights.append(item.weight.detach().numpy())
            action_net_biases.append(item.bias.detach().numpy())
        elif isinstance(item, th.nn.modules.activation.Tanh):
            action_net_activation_fns.append(np.tanh)
        else:
            raise NotImplementedError
elif isinstance(model.policy.action_net, th.nn.modules.linear.Linear):
    action_net_weights.append(model.policy.action_net.weight.detach().numpy())
    action_net_biases.append(model.policy.action_net.bias.detach().numpy())
else:
    raise NotImplementedError

# for i in range(len(policy_net_architecture)):
#     policy_net_weights.append(np.array(model.policy.state_dict()[f"mlp_extractor.policy_net.{2*i}.weight"]))
#     policy_net_biases.append(np.array(model.policy.state_dict()[f"mlp_extractor.policy_net.{2*i}.bias"]))

# if len(action_net_architecture) == 1:
#     action_net_weights.append(np.array(model.policy.state_dict()[f"action_net.weight"]))
#     action_net_biases.append(np.array(model.policy.state_dict()[f"action_net.bias"]))
# else:
#     raise NotImplementedError
#     for i in range(len(action_net_architecture)):
#         action_net_weights.append(np.array(model.policy.state_dict()[f"action_net.{2*i}.weight"]))
#         action_net_biases.append(np.array(model.policy.state_dict()[f"action_net.{2*i}.bias"]))

# activation_fn_class = model.policy.activation_fn

# if activation_fn_class.__name__ == "Tanh":
#     act_fn = np.tanh
# else:
#     raise NotImplementedError

def policy_net(layer):
    for i in range(len(policy_net_weights)):
        layer = policy_net_weights[i] @ layer + policy_net_biases[i]
        layer = policy_net_activation_fns[i](layer)
    return layer

def action_net(layer):
    for i in range(len(action_net_weights)):
        layer = action_net_weights[i] @ layer + action_net_biases[i]
        if i < len(action_net_activation_fns):
            layer = action_net_activation_fns[i](layer)
    return layer

def NN(obs):
    obs = np.array(obs)
    layer = obs
    layer = policy_net(layer)
    layer = action_net(layer)   
    
    return layer

obs = [1,1,1,1,1.0]
obs_t = model.policy.obs_to_tensor(obs)[0]
features = model.policy.extract_features(obs_t)
NN(obs)
model.predict(obs, deterministic=True)
assert sum(NN(obs) - model.predict(obs, deterministic=True)[0]) < 1e-4

print("done")