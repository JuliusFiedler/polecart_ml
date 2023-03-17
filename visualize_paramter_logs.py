import numpy as np
import os
import torch as th
import pickle
import matplotlib.pyplot as plt
import pygame
from pygame import gfxdraw
from ipydex import IPS, activate_ips_on_exception

from util import *

activate_ips_on_exception()

model_name = "CartPoleContinousSwingupEnv___2023_03_16__10_46_05"

path = os.path.join(ROOT_PATH, "models", model_name, "parameter_log.pickle")

with open(path, "rb") as f:
    data = pickle.load(f)

policy_net_w_1 = []
policy_net_w_2 = []
policy_net_b_1 = []
policy_net_b_2 = []
action_net_w = []
action_net_b = []

for d in data:
    policy_net_w_1.append(d["policy"]["mlp_extractor.policy_net.0.weight"].detach().numpy())
    policy_net_w_2.append(d["policy"]["mlp_extractor.policy_net.2.weight"].detach().numpy())
    policy_net_b_1.append(d["policy"]["mlp_extractor.policy_net.0.bias"].detach().numpy())
    policy_net_b_2.append(d["policy"]["mlp_extractor.policy_net.2.bias"].detach().numpy())
    action_net_w.append(d["policy"]["action_net.weight"].detach().numpy())
    action_net_b.append(d["policy"]["action_net.bias"].detach().numpy())

all_data = [
    policy_net_w_1,
    policy_net_b_1,
    policy_net_w_2,
    policy_net_b_2,
    action_net_w,
    action_net_b,
]
pygame.init()

# display
os.environ["SDL_VIDEO_WINDOW_POS"] = f"{2000},{400}"
pygame.display.set_caption("Parameter Visualization")
screen_width = 1000
screen_height = 800
game_display = pygame.display.set_mode((screen_width, screen_height))

# init stuff
clock = pygame.time.Clock()


def visualize_layer(layer, offset_l, offset_t, base_color):
    for i, p in np.ndenumerate(layer):
        if len(i) == 2:
            ir, ic = i
        else:
            ir, ic = i[0], 0
        l = offset_l + ic * block_size[0]
        r = l + block_size[0]
        t = offset_t + ir * block_size[1]
        b = t + block_size[1]
        coords = [(l, b), (l, t), (r, t), (r, b)]

        # assume all weights are in [-1, 1]
        assert np.abs(p) <= 2, "rethink color scaling!"
        v = 4
        if v == 1:
            color_value = int((1 + p / 2) * 255 / 2)
            color = [0, 0, 0]
            color[base_color] = color_value
        elif v == 2:
            color_value = int((p / 2 + 1) / 2 * 3 * 255)
            if color_value <= 255:
                color = [color_value, 0, 0]
            elif color_value <= 2 * 255:
                color = [255, color_value - 255, 0]
            else:
                color = [255, 255, color_value - 2 * 255]
        elif v == 3:
            color_value = int(np.abs(p) / 2 * 255)
            if p <= 0:
                color = [color_value, 0, 0]
            else:
                color = [0, color_value, 0]
        elif v == 4:
            color_value = int(np.abs(p / 2) * 3 * 255)
            if color_value <= 255:
                color = [color_value, 0, 0]
            elif color_value <= 2 * 255:
                color = [255, color_value - 255, 0]
            else:
                color = [255, 255, color_value - 2 * 255]

        assert all(np.array(color) <= 255)
        gfxdraw.filled_polygon(surf, coords, tuple(color))


for i in range(len(data)):
    surf = pygame.Surface((screen_width, screen_height))
    surf.fill((255, 255, 255))

    text_to_screen(surf, f"Update Step {i}", (screen_width - 70, 30))
    block_size = (10, 10)
    offset_l = 10
    offset_t = 10

    for k, layer_list in enumerate(all_data):
        if len(layer_list[i].shape) == 2:
            if layer_list[i].shape[0] < layer_list[i].shape[1]:
                layer_list[i] = layer_list[i].T
        visualize_layer(layer_list[i], offset_l, offset_t, k % 3)
        if len(layer_list[i].shape) == 2:
            off_l = layer_list[i].shape[1] * block_size[0]
        else:
            off_l = block_size[0]
        offset_l += off_l + 2

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    game_display.blit(surf, (0, 0))
    clock.tick(10)
    pygame.display.update()

IPS()
