import pygame as pg
import numpy as np

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
red = (255, 0, 0)


def text_to_screen(surf, text, pos, font=None, fontsize=16, color=None):
    if font is None:
        font = pg.font.Font("freesansbold.ttf", fontsize)
    if color is None:
        color = black
    obj = font.render(text, True, color, white)
    obj_rect = obj.get_rect()
    obj_rect.center = (int(pos[0]), int(pos[1]))
    surf.blit(obj, obj_rect)
    
    