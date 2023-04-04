import pygame as pg
import numpy as np
import os
from colorama import Style, Fore
import datetime

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

white = (255, 255, 255)
grey_1 = (200, 200, 200)
grey_2 = (100, 100, 100)
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


def text_to_screen(surf, text, pos, font=None, fontsize=16, color=None, rotation=0, return_rect=False):
    if font is None:
        font = pg.font.Font("freesansbold.ttf", fontsize)
    if color is None:
        color = black
    obj = font.render(text, True, color, white)
    obj = pg.transform.rotate(obj, rotation)
    obj_rect = obj.get_rect()
    obj_rect.center = (int(pos[0]), int(pos[1]))
    surf.blit(obj, obj_rect)
    if return_rect:
        return obj_rect


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


class Slider:
    def __init__(self, disp, x, y, w=50, h=20, value=0, value_range=[-1, 1]) -> None:
        self.disp = disp
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.value = value
        self.value_range = value_range
        self.rect = pg.Rect(self.x, self.y, self.w, self.h)
        self.zero_button = Button(
            self.disp, self.x, self.y + self.h + 2, self.w, self.h, grey_2, grey_1, "zero", action=self.zero
        )

    def show(self):
        pg.draw.rect(self.disp, black, self.rect, width=2)
        pos = (
            self.x + self.value / (self.value_range[-1] - self.value_range[0]) * self.w + self.w / 2,
            self.y + self.h / 2,
        )
        pg.draw.circle(self.disp, color=blue, center=pos, radius=5)
        self.zero_button.show()
        text_to_screen(
            self.disp, str(round(self.value, 3)), (self.x + self.w / 2, self.y + self.h * 2 + 4 + self.h / 2)
        )

    def update(self, event_list):
        for event in event_list:
            if event.type == pg.MOUSEBUTTONDOWN:
                mouse_pos = pg.mouse.get_pos()
                if self.rect.collidepoint(mouse_pos):
                    self.value = ((mouse_pos[0] - self.x) / self.w - 0.5) * (self.value_range[-1] - self.value_range[0])

    def zero(self):
        self.value = 0


class Node:
    def __init__(self, disp, x, y, r, value=0, color=black):
        self.disp = disp
        self.x = x
        self.y = y
        self.r = r
        self.color = color
        self.value = value

    def show(self):
        try:
            pg.draw.circle(self.disp, self.color, (self.x, self.y), self.r)
        except ValueError:
            pg.draw.circle(self.disp, black, (self.x, self.y), self.r)


class Connection:
    def __init__(self, surf, start, end, color=black) -> None:
        self.surf = surf
        self.start = start
        self.end = end
        self.color = color

    def show(self):
        if isinstance(self.start, (tuple, list, np.ndarray)):
            start = tuple(self.start)
        elif isinstance(self.start, Node):
            start = (self.start.x, self.start.y)
        if isinstance(self.end, (tuple, list, np.ndarray)):
            end = tuple(self.end)
        elif isinstance(self.end, Node):
            end = (self.end.x, self.end.y)
        pg.draw.line(self.surf, self.color, start_pos=start, end_pos=end)


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
