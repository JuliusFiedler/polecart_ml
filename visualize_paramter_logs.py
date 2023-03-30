import numpy as np
import os, sys
import torch as th
import pickle
import matplotlib.pyplot as plt
import pygame
from pygame import gfxdraw
from ipydex import IPS, activate_ips_on_exception

from util import *

activate_ips_on_exception()

model_name = "CartPoleContinous2Env___2023_03_29__15_49_42"

path = os.path.join(ROOT_PATH, "models", model_name, "parameter_log.pickle")

with open(path, "rb") as f:
    data = pickle.load(f)

keys = list(data[0]["policy"].keys())
if "log_std" in keys:
    keys.remove("log_std")
max_value = 0
data_dict = {}
for k in keys:
    data_dict[k] = []
for d in data:
    for k in keys:
        data_dict[k].append(d["policy"][k].detach().numpy())
        # filter for biggest (abs) parameter for color scaling later
        m = np.max(np.abs(d["policy"][k].detach().numpy()))
        if max_value < m:
            max_value = m
print("max abs parameter", max_value)

num_updates = len(data_dict[keys[0]])
# v_line = 1630208 / 5001216
v_line = None
if v_line is not None:
    at = int(v_line * num_updates)
    for k in keys:
        for p in data_dict[k][at]:
            print(at, k, p)
        for p in data_dict[k][-1]:
            print("end", k, p)

fig, ax = plt.subplots(len(keys), 1)
for idx, k in enumerate(keys):
    t = np.arange(len(data_dict[k]))
    y = np.array(data_dict[k])
    if len(y.shape) == 3:
        y = y[:,0,:]
    assert len(y.shape) == 2
    for i in range(y.shape[1]):
        ax[idx].plot(t, y[:,i], label=i)
        ax[idx].set_title(k)
        ax[idx].legend()
        if v_line is not None:
            ax[idx].vlines([v_line*y.shape[0]], min(y[:,i]), max(y[:,i]), color="k")
plt.show()


    

pygame.init()

# display
os.environ["SDL_VIDEO_WINDOW_POS"] = f"{2000},{400}"
pygame.display.set_caption("Parameter Visualization")
screen_width = 1000
screen_height = 800
game_display = pygame.display.set_mode((screen_width, screen_height))

# init stuff
clock = pygame.time.Clock()


def visualize_layer(layer, offset_l, offset_t, base_color, max_value):
    desc = ""
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
        assert np.abs(p) <= max_value, "rethink color scaling!"
        v = 4
        if v == 1:
            desc = "1 color per layer, brighter means >0, darker means <0"
            color_value = int((1 + p / max_value) * 255 / 2)
            color = [0, 0, 0]
            color[base_color] = color_value
        elif v == 2:
            desc = "black = -max_value, white = max_value, continous scaling inbetween"
            color_value = int((p / max_value + 1) / 2 * 3 * 255)
            if color_value <= 255:
                color = [color_value, 0, 0]
            elif color_value <= 2 * 255:
                color = [255, color_value - 255, 0]
            else:
                color = [255, 255, color_value - 2 * 255]
        elif v == 3:
            desc = "red means <0, green means >0, dark = 0, bright = high"
            color_value = int(np.abs(p) / max_value * 255)
            if p <= 0:
                color = [color_value, 0, 0]
            else:
                color = [0, color_value, 0]
        elif v == 4:
            desc = "absolute values, black = 0, brighter = higher"
            color_value = int(np.abs(p / max_value) * 3 * 255)
            if color_value <= 255:
                color = [color_value, 0, 0]
            elif color_value <= 2 * 255:
                color = [255, color_value - 255, 0]
            else:
                color = [255, 255, color_value - 2 * 255]

        assert all(np.array(color) <= 255)
        gfxdraw.filled_polygon(surf, coords, tuple(color))
    text_to_screen(surf, desc, (300, 700))


for i in range(len(data)):
    surf = pygame.Surface((screen_width, screen_height))
    surf.fill((255, 255, 255))

    text_to_screen(surf, repr(keys), (300, 720))
    text_to_screen(surf, f"Update Step {i}", (screen_width - 70, 30))
    block_size = (10, 10)
    offset_l = 10
    offset_t = 10

    # for k, layer_list in enumerate(all_data):
    for k, layer_list in enumerate(data_dict.values()):
        if len(layer_list[i].shape) == 2:
            if layer_list[i].shape[0] < layer_list[i].shape[1]:
                layer_list[i] = layer_list[i].T
        visualize_layer(layer_list[i], offset_l, offset_t, k % 3, max_value)
        if len(layer_list[i].shape) == 2:
            off_l = layer_list[i].shape[1] * block_size[0]
        else:
            off_l = block_size[0]
        offset_l += off_l + 2

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
    game_display.blit(surf, (0, 0))
    clock.tick(50)
    pygame.display.update()

IPS()
