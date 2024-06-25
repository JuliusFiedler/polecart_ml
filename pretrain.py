import csv
import numpy as np
from ipydex import IPS, activate_ips_on_exception
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ppo.supervised import PretrainedNet

activate_ips_on_exception()

with open("trajectories/cartpole_swingup_state_and_action.csv", newline="") as csvfile:
    actions = []
    states = []
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        actions.append(row[-1])
        states.append(row[:-1])


states = torch.from_numpy(np.array(states, dtype=float))
actions = torch.from_numpy(np.array(actions, dtype=float))


net = PretrainedNet()
criterion = nn.MSELoss(reduction="sum")
# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=1e-6)
net = net.float()

for epoch in range(2000):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(zip(states, actions), 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterion(outputs.float(), labels.float())
        # zero the parameter gradients
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:  # print every 2000 mini-batches
            print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}")
            running_loss = 0.0
torch.save(net.state_dict(), "trajectories/swingupNN.pth")
print("Finished Training")
IPS()
