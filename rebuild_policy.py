import numpy as np
import torch as th
from ipydex import IPS, activate_ips_on_exception
import sympy as sp
import symbtools as st

from ppo.ppo_agent import PPOAgent
from envs.cartpole import CartPoleContinous2Env

activate_ips_on_exception()

env = CartPoleContinous2Env()

agent = PPOAgent(env)

agent.load_model("cartpole_model__CartPoleContinous2Env___2023_03_02__16_40_49.h5")
model = agent.model

# input_features = 4
# output_features = 1

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

# TODO: maybe this should have the form of a dict?


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
        # TODO: this is not generalizable
    return layer


def NN(obs):
    obs = np.array(obs)
    layer = obs
    layer = policy_net(layer)
    layer = action_net(layer)

    return layer


obs = [0.01, 0.01, 0.01, 0.01]
obs_t = model.policy.obs_to_tensor(obs)[0]
features = model.policy.extract_features(obs_t)
NN(obs)
model.predict(obs, deterministic=True)
# this doesnt work if model clips action and NN does not, use sensible obs
assert sum(NN(obs) - model.predict(obs, deterministic=True)[0]) < 1e-4


# linearization of NN

## scalar method
# Bpi = [sp.var(f"Bp{i}") for i in range(len(policy_net_biases))]
# Wpi = [sp.var(f"Wp{i}") for i in range(len(policy_net_weights))]
# Bai = [sp.var(f"Ba{i}") for i in range(len(action_net_biases))]
# Wai = [sp.var(f"Wa{i}") for i in range(len(action_net_weights))]
# x = sp.var("x")

# # f = Bai[0] + Wai[0] * sp.tanh(Bpi[1] + Wpi[1] * sp.tanh(Bpi[0] + Wpi[0] * x))
# f = Bai[0] + Wai[0] * sp.tanh(Bpi[0] + Wpi[0] * x)
# J = sp.diff(f, x)
# J0 = J.subs(x, 0)
# # ! unclear if and how dimensions would work
# IPS()

## Matrix method
Bpi = [sp.MatrixSymbol(f"Bp{i}", M.shape[0], 1) for i, M in enumerate(policy_net_biases)]
Wpi = [sp.MatrixSymbol(f"Wp{i}", *M.shape) for i, M in enumerate(policy_net_weights)]
Bai = [sp.MatrixSymbol(f"Ba{i}", M.shape[0], 1) for i, M in enumerate(action_net_biases)]
Wai = [sp.MatrixSymbol(f"Wa{i}", *M.shape) for i, M in enumerate(action_net_weights)]

assert all([np.tanh == i for i in policy_net_activation_fns]), "Not Implemented"
policy_net_activation_fns_sp = [sp.tanh for i in policy_net_activation_fns]


x = sp.Matrix([sp.var(f"x{i}") for i in range(env.observation_space.shape[0])])
f = x

for i in range(len(Wpi)):
    f = (Wpi[i] @ f + Bpi[i]).applyfunc(policy_net_activation_fns_sp[i])
for i in range(len(Wai)):
    f = Wai[i] @ f + Bai[i]

IPS()
J = st.jac(f, x)

J0 = J.subs([*zip(x, sp.zeros(len(x)))])

subslist = []
subslist.extend([(Bpi[i], sp.Matrix(policy_net_biases[i])) for i in range(len(Bpi))])
subslist.extend([(Wpi[i], sp.Matrix(policy_net_weights[i])) for i in range(len(Wpi))])
subslist.extend([(Bai[i], sp.Matrix(action_net_biases[i])) for i in range(len(Bai))])
subslist.extend([(Wai[i], sp.Matrix(action_net_weights[i])) for i in range(len(Wai))])

#! substitution of Matrices only works with sp.Matrix, not with np array
K = J0.subs(subslist).doit()
print(K)
#! evaluation of subbed matrix can be done by calling .doit()
IPS()

print("done")
