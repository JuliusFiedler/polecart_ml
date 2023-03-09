import pygame as pg
import numpy as np
import os
from colorama import Style, Fore
import datetime

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

white = (255, 255, 255)
black = (0, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 200)
red = (255, 0, 0)
red = (200, 0, 0)
light_red = (255, 0, 0)
light_blue = (0, 0, 255)


def bright(txt):
    return f"{Style.BRIGHT}{txt}{Style.RESET_ALL}"


def bgreen(txt):
    return f"{Fore.GREEN}{Style.BRIGHT}{txt}{Style.RESET_ALL}"


def bred(txt):
    return f"{Fore.RED}{Style.BRIGHT}{txt}{Style.RESET_ALL}"


def yellow(txt):
    return f"{Fore.YELLOW}{txt}{Style.RESET_ALL}"


def text_to_screen(surf, text, pos, font=None, fontsize=16, color=None):
    if font is None:
        font = pg.font.Font("freesansbold.ttf", fontsize)
    if color is None:
        color = black
    obj = font.render(text, True, color, white)
    obj_rect = obj.get_rect()
    obj_rect.center = (int(pos[0]), int(pos[1]))
    surf.blit(obj, obj_rect)


def text_objects(msg, font, text_color=black):
    text_surface = font.render(msg, True, text_color)
    return text_surface, text_surface.get_rect()


class Button:
    def __init__(
        self,
        disp,
        x,
        y,
        w,
        h,
        inactive_color,
        active_color,
        text,
        font=None,
        text_color=black,
        action=None,
        action_args=[],
    ) -> None:
        self.disp = disp
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.inactive_color = inactive_color
        self.active_color = active_color
        self.text = text
        self.font = pg.font.Font("freesansbold.ttf", 16)
        self.text_color = text_color
        self.action = action
        self.already_pressed = False
        self.action_args = action_args

    def show(self):
        mouse = pg.mouse.get_pos()
        click = pg.mouse.get_pressed()
        button_pressed = False
        button_pos = (self.x, self.y)
        button_size = button_2_size = (self.w, self.h)
        if (
            mouse[0] > button_pos[0]
            and mouse[0] < button_pos[0] + button_size[0]
            and mouse[1] > button_pos[1]
            and mouse[1] < button_pos[1] + button_size[1]
        ):
            pg.draw.rect(self.disp, self.active_color, button_pos + button_size)
            if click[0] and self.action != None:
                # Button entprellen
                if not self.already_pressed:
                    self.action(*self.action_args)
                    self.already_pressed = True
            elif not click[0] and self.already_pressed:
                self.already_pressed = False

        else:
            pg.draw.rect(self.disp, self.inactive_color, button_pos + button_size)

        text_surface, text_rect = text_objects(self.text, self.font, self.text_color)
        text_rect.center = (button_pos[0] + button_size[0] / 2, button_pos[1] + button_size[1] / 2)
        self.disp.blit(text_surface, text_rect)


def project_to_interval(state, min=-np.pi, max=np.pi):
    """map the state to the given interval, specifically map theta to [-pi, pi]"""
    assert abs(min) + abs(max) == 2 * np.pi, "interval is not of size 2*pi"

    if isinstance(state, (np.ndarray, list, tuple)):
        state = np.array(state, dtype=float)
        angle = state[2]
    else:
        angle = state

    angle = angle % (2 * np.pi)
    if angle < min:
        angle += np.pi * 2
    elif angle > max:
        angle -= np.pi * 2

    assert min <= angle
    assert angle <= max

    if isinstance(state, np.ndarray):
        state[2] = angle
    else:
        state = angle

    return state


# based on
# source: https://stackoverflow.com/a/46928226/333403
# by chidimo
def smooth_timedelta(start_datetime, end_datetime=None):
    """Convert a datetime.timedelta object into Days, Hours, Minutes, Seconds."""
    if end_datetime is None:
        end_datetime = datetime.datetime.now()
    timedeltaobj = end_datetime - start_datetime
    secs = timedeltaobj.total_seconds()
    timetot = ""
    if secs > 86400:  # 60sec * 60min * 24hrs
        days = secs // 86400
        timetot += "{}d".format(int(days))
        secs = secs - days * 86400

    if secs > 3600:
        hrs = secs // 3600
        timetot += " {}h".format(int(hrs))
        secs = secs - hrs * 3600

    if secs > 60:
        mins = secs // 60
        timetot += " {}m".format(int(mins))
        secs = secs - mins * 60

    if secs > 0:
        timetot += " {}s".format(int(secs))
    return timetot
