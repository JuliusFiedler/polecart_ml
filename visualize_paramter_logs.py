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

# model_name = "CartPoleContinous2Env___2023_03_16__11_03_00_good"
model_name = "CartPoleContinous2Env___2023_04_04__15_22_29"
model_name_compare = "CartPoleContinous2Env___2023_04_04__11_10_59__max_tr_1000__x_off"

# path = os.path.join(ROOT_PATH, "models", model_name, "parameter_log.pickle")


pygame.init()

# display
os.environ["SDL_VIDEO_WINDOW_POS"] = f"{1920},{25}"
pygame.display.set_caption("Parameter Visualization")
screen_width = 1920
screen_height = 1175
game_display = pygame.display.set_mode((screen_width, screen_height))

# init stuff
clock = pygame.time.Clock()
compare_button = None
evolution_button = None
nodes_button = None
backwards_button = None
back_button = None
slider_1 = None


def import_data(name):
    path = os.path.join(ROOT_PATH, "models", name, "training_logs.p")

    with open(path, "rb") as f:
        logs = pickle.load(f)
    data = logs["NN_updates"]

    max_value = 0
    data_dict = data["policy"]
    keys = list(data["policy"].keys())
    if "log_std" in keys:
        keys.remove("log_std")
        data_dict.pop("log_std")

    # filter for biggest (abs) parameter for color scaling later
    for k in keys:
        for entry in data_dict[k]:
            m = np.max(np.abs(entry))
            if max_value < m:
                max_value = m
    print("max abs parameter", max_value)
    return keys, data_dict, max_value, logs


keys, data_dict, max_value, logs = import_data(model_name)

# action net of trained model
action_net_dict = {}
for i, key in enumerate(keys):
    if f"policy_net" in key and "weight" in key:
        layer = int(int(key.split(".")[2]) / 2)
        action_net_dict[layer] = {
            "w": data_dict[f"mlp_extractor.policy_net.{2 * layer}.weight"][-1],
            "b": data_dict[f"mlp_extractor.policy_net.{2 * layer}.bias"][-1],
            "name": key,
        }
    elif "action_net" in key and "weight" in key:
        layer = len(action_net_dict.keys())
        action_net_dict[layer] = {
            "w": data_dict[f"action_net.weight"][-1],
            "b": data_dict[f"action_net.bias"][-1],
            "name": key,
        }

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
block_size = (10, 10)


def visualize_layer(surf, layer, offset_l, offset_t, base_color, max_value, v=4):
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

        assert np.abs(p) <= max_value, "rethink color scaling!"
        if max_value == 0:
            max_value = 1
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


def parameter_evolution():
    for i in range(num_updates):
        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))

        text_to_screen(surf, f"Update Step {i}", (screen_width - 70, 30))
        offset_l = 10
        offset_t = 10

        # for k, layer_list in enumerate(all_data):
        for k, layer_list in enumerate(data_dict.values()):
            if len(layer_list[i].shape) == 2:
                if layer_list[i].shape[0] < layer_list[i].shape[1]:
                    layer_list[i] = layer_list[i].T
            visualize_layer(surf, layer_list[i], offset_l, offset_t, k % 3, max_value)
            text_to_screen(surf, keys[k], (offset_l + 5, 770), fontsize=12, rotation=90)
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


def get_color(value, min_v, max_v):
    v = 1
    if v == 0:
        scaled_v = value / (max_v - min_v) * 255
    elif v == 1:
        scaled_v = np.abs(value) / max(np.abs([min_v, max_v])) * 255

    color = (scaled_v, 0, 0)
    return color


def visualize_nodes():
    surf = pygame.Surface((screen_width, screen_height))
    num_inputs = 4
    sliders = [None for i in range(num_inputs)]
    back_button = None
    done = False
    while not done:
        surf.fill((255, 255, 255))
        nodes = {}
        k = 0
        r = 7
        offset_l = 50
        offset_t = 10

        # Buttons and Sliders
        def back():
            global done
            done = True

        if back_button is None:
            back_button = Button(game_display, 1800, 1100, 50, 20, red, blue, "Back", action=back)
        eventlist = pg.event.get()
        obs = np.zeros(num_inputs)
        for i in range(num_inputs):
            if sliders[i] is None:
                sliders[i] = Slider(surf, 20, offset_t + 10 + i * screen_height / num_inputs, value=0)
            sliders[i].update(eventlist)
            sliders[i].show()
            obs[i] = sliders[i].value

        layer_values = obs
        layer_values_dict = {}
        layer_values_dict[0] = layer_values
        for i, key in enumerate(action_net_dict.keys()):
            layer_values = action_net_dict[key]["w"] @ layer_values + action_net_dict[key]["b"]
            layer_values_dict[i + 1] = layer_values

        # input layer
        nodes[k] = []
        for i in range(num_inputs):
            value = layer_values_dict[k][i]
            color = get_color(value, min(layer_values_dict[k]), max(layer_values_dict[k]))
            n = Node(surf, offset_l, offset_t, r, value=value, color=color)
            n.show()
            nodes[k].append(n)
            # inputs.append(InputBox(surf, 50, offset_t + 40, 50, default=0.1))
            offset_t += (screen_height - 20) / num_inputs
        k += 1
        offset_l += 600
        # all other layers
        for key in data_dict.keys():
            offset_t = 10
            if ("mlp_extractor.policy_net" in key or "action_net" in key) and "bias" in key:
                nodes[k] = []
                num_nodes = len(data_dict[key][-1])
                for i in range(num_nodes):
                    value = layer_values_dict[k][i]
                    color = get_color(value, min(layer_values_dict[k]), max(layer_values_dict[k]))
                    n = Node(surf, offset_l, offset_t, r, value=value, color=color)
                    if n.color[0] > 255 / 2:
                        n.show()
                    nodes[k].append(n)
                    offset_t += (screen_height - 20) / num_nodes
                k += 1
                offset_l += 600

        # for key in list(nodes.keys())[:-1]:
        #     for s in nodes[key]:
        #         for e in nodes[key+1]:
        #             Connection(surf, s, e, color=(200, 200, 200))#.show()

        pg.event.get()

        game_display.blit(surf, (0, 0))
        back_button.show()
        clock.tick(50)
        pygame.display.update()


def visualize_backwards():
    surf = pygame.Surface((screen_width, screen_height))
    num_inputs = 4
    sliders = [None for i in range(num_inputs)]
    back_button = None
    done = False
    while not done:
        surf.fill((255, 255, 255))
        nodes = {}
        k = 0
        r = 7
        offset_l = 50
        offset_t = 10

        # Buttons and Sliders
        def back():
            global done
            done = True

        if back_button is None:
            back_button = Button(game_display, 1800, 1100, 50, 20, red, blue, "Back", action=back)
        eventlist = pg.event.get()
        obs = np.zeros(num_inputs)
        for i in range(num_inputs):
            if sliders[i] is None:
                sliders[i] = Slider(surf, 20, offset_t + 10 + i * screen_height / num_inputs, value=0)
            sliders[i].update(eventlist)
            sliders[i].show()
            obs[i] = sliders[i].value

        layer_values = obs
        layer_values_dict = {}
        layer_values_dict[0] = layer_values
        for i, key in enumerate(action_net_dict.keys()):
            layer_values = action_net_dict[key]["w"] @ layer_values + action_net_dict[key]["b"]
            layer_values_dict[i + 1] = layer_values

        # input layer
        nodes[k] = []
        for i in range(num_inputs):
            value = layer_values_dict[k][i]
            color = get_color(value, min(layer_values_dict[k]), max(layer_values_dict[k]))
            n = Node(surf, offset_l, offset_t, r, value=value, color=color)
            n.show()
            nodes[k].append(n)
            # inputs.append(InputBox(surf, 50, offset_t + 40, 50, default=0.1))
            offset_t += (screen_height - 20) / num_inputs
        k += 1
        offset_l += 600
        # all other layers
        for key in data_dict.keys():
            offset_t = 10
            if ("mlp_extractor.policy_net" in key or "action_net" in key) and "bias" in key:
                nodes[k] = []
                num_nodes = len(data_dict[key][-1])
                for i in range(num_nodes):
                    value = layer_values_dict[k][i]
                    index = np.argmax(np.abs(layer_values_dict[k]))
                    color = get_color(value, min(layer_values_dict[k]), max(layer_values_dict[k]))
                    n = Node(surf, offset_l, offset_t, r, value=value, color=color)
                    if n.color[0] > 255 * 0.9:
                        n.show()
                    nodes[k].append(n)
                    offset_t += (screen_height - 20) / num_nodes
                k += 1
                offset_l += 600

        # for key in list(nodes.keys())[:-1]:
        #     for s in nodes[key]:
        #         for e in nodes[key+1]:
        #             Connection(surf, s, e, color=(200, 200, 200))#.show()

        pg.event.get()

        game_display.blit(surf, (0, 0))
        back_button.show()
        clock.tick(50)
        pygame.display.update()


def compare_parameters(idx=-1):
    done = False
    keys_2, data_dict_2, _, logs_2 = import_data(model_name_compare)
    assert keys == keys_2, "comparison between differently structured NNs"
    while not done:
        if np.abs(idx) >= len(data_dict[keys[0]]):
            idx = -1
            print("max index", len(data_dict[keys[0]]))

        comp_dict = {}
        max_v = 0
        for key in keys:
            assert (
                data_dict[key][idx].shape == data_dict_2[key][idx].shape
            ), "comparison between differently structured NNs"
            comp_dict[key] = data_dict[key][idx] - data_dict_2[key][idx]
            m = np.max(np.abs(comp_dict[key]))
            if max_v < m:
                max_v = m
        surf = pygame.Surface((screen_width, screen_height))
        surf.fill((255, 255, 255))

        offset_l = 10
        offset_t = 10
        text_to_screen(surf, f"Episode {logs['NN_updates']['episode'][idx]}", (100, 900))
        text_to_screen(surf, f"Step {logs['NN_updates']['step'][idx]}", (100, 920))

        for k, layer in enumerate(comp_dict.values()):
            if len(layer.shape) == 2:
                if layer.shape[0] < layer.shape[1]:
                    layer = layer.T
            visualize_layer(surf, layer, offset_l, offset_t, k % 3, max_v, v=3)
            text_to_screen(surf, keys[k], (offset_l + 5, 770), fontsize=12, rotation=90)
            if len(layer.shape) == 2:
                off_l = layer.shape[1] * block_size[0]
            else:
                off_l = block_size[0]
            offset_l += off_l + 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        game_display.blit(surf, (0, 0))
        clock.tick(50)
        pygame.display.update()
        inp = input("new index (int) or back [n]")
        if inp == "n":
            done = True
        else:
            try:
                idx = int(inp)
            except:
                pass


done = False
while not done:
    game_display.fill((255, 255, 255))
    if compare_button is None:
        compare_button = Button(
            game_display, 50, 50, 250, 20, red, light_red, "Compare Agents", action=compare_parameters
        )
    compare_button.show()
    if evolution_button is None:
        evolution_button = Button(
            game_display, 50, 80, 250, 20, red, light_red, "Parameter Evolution", action=parameter_evolution
        )
    evolution_button.show()
    if nodes_button is None:
        nodes_button = Button(game_display, 50, 110, 250, 20, red, light_red, "Visualize Nodes", action=visualize_nodes)
    nodes_button.show()
    if backwards_button is None:
        backwards_button = Button(
            game_display, 50, 140, 250, 20, red, light_red, "Visualize Backwards", action=visualize_backwards
        )
    backwards_button.show()

    for ev in pygame.event.get():
        if ev.type == pygame.QUIT:
            IPS()
            pygame.quit()
            done = True

    clock.tick(50)
    pygame.display.update()
