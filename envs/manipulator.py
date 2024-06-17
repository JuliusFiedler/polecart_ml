import math
from typing import Optional, Union
import os
import numpy as np
from scipy.integrate import solve_ivp
from ipydex import IPS
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

import util as u

class ManipulatorEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    def __init__(self) -> None:
        self.name = self.__class__.__name__
        self.eta = 1.42
        self.a1 = 0.18
        self.a2 = 20
        self.tau = 0.02
        self.ep_step_count = 0
        self.total_step_count = 0
        self.episode_count = 0

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None

        self.clock = None
        self.isopen = True
        self.state = None
        self.action = None
        self.reward = None
        self.target = None


        high = np.array(
            [
                np.finfo(np.float32).max,  # x
                np.finfo(np.float32).max,  # xdot
                np.finfo(np.float32).max,  # phi
                np.finfo(np.float32).max,  # phidot
                2,                         # target x
                2,                         # target y
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(-10, 10, (1,), float)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.request_reset = False
        self.reset_button = None  # initialize some objects, that have to be persistent and not be recreated each step

        import envs.parameter.ManipulatorEnv as c
        self.c = c

        super().__init__()

    def calc_new_state(self, action):
        if hasattr(action, "shape"):
            u = action[0]
        else:
            u = action
        def rhs(t, x):
            phi1, phi2, omega1, omega2 = x
            # x1... phi1, angle at active joint
            # x2... phi2, angle at passive joint
            # x3... omega1, velocity of first arm
            # x4... omega2, velocity of second arm

            dxdt1 = omega1
            dxdt2 = omega2
            dxdt3 = u
            dxdt4 = -u * (1 + self.eta*np.cos(phi2)) - self.eta*omega1**2*np.sin(phi2) #- self.a1 * np.arctan(self.a2 * x[3])
            return np.array([dxdt1, dxdt2, dxdt3, dxdt4])

        tt = np.linspace(0, self.tau, 2)
        xx0 = np.array(self.state).flatten()
        s = solve_ivp(rhs, (0, self.tau), xx0, t_eval=tt)

        # for i in range(4):
        #     plt.plot(s.t, s.y[i], label=f"$x_{i}$")
        # plt.legend()
        # plt.show()
        # q1, q2, q1dot, q2dot
        self.state = (s.y[:, -1].flatten())
        return self.state

    def step(self, action):
        self.ep_step_count += 1
        self.total_step_count += 1
        if not self.action_space.contains(action):
            action = np.array([action], dtype=float)
        self.action = action
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        old_state = self.state

        self.state = self.calc_new_state(action)

        # Debug reproducibility
        # with open("test.txt", "a") as f:
        #     f.write(f"\nStep. {self.total_step_count}, old State: {old_state}, Action: {action}, new State: {self.state}")

        self.reward, terminated, truncated, info = self.get_reward()

        if self.render_mode == "human":
            self.render()

        if self.request_reset:
            truncated = True
        # self.save_step_data(state, action, self.reward, terminated, truncated, info)

        state = self.state

        return state, self.reward, terminated, truncated, info

    def get_reward(self):
        return self.c.get_reward(self)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled("pygame is not installed, run `pip install gymnasium[classic-control]`") from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                # TODO does this work if there is only one screen?
                os.environ["SDL_VIDEO_WINDOW_POS"] = f"{2000},{400}"
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        polewidth = 10.0
        polelength = 100

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        center_coord = (self.screen_width/2, self.screen_height/2)
        join_coord = np.array((np.cos(x[0])*polelength, -np.sin(x[0])*polelength)) + np.array(center_coord)
        end_coord =  np.array((np.cos(x[0]+x[1])*polelength, -np.sin(x[0]+x[1])*polelength)) + np.array(join_coord)

        # first arm
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelength,
            0,
        )
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[0] - np.pi/2)
            coord = (coord[0] + center_coord[0], coord[1] + center_coord[1])
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        # second arm
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelength,
            0,
        )
        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[0] - x[1] - np.pi/2)
            coord = (coord[0] + join_coord[0], coord[1] + join_coord[1])
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        # center joint
        gfxdraw.filled_circle(
            self.surf,
            int(center_coord[0]),
            int(center_coord[1]),
            int(polewidth / 2),
            (129, 132, 203),
        )
        # middle joint
        gfxdraw.filled_circle(
            self.surf,
            int(join_coord[0]),
            int(join_coord[1]),
            int(polewidth / 2),
            (129, 132, 203),
        )
        # end joint
        gfxdraw.filled_circle(
            self.surf,
            int(end_coord[0]),
            int(end_coord[1]),
            int(polewidth / 2),
            (129, 132, 203),
        )

        # target
        if self.target is not None:
            gfxdraw.filled_circle(
                self.surf,
                int(self.target[0]*polelength*2+center_coord[0]),
                int(self.target[1]*polelength*2+center_coord[1]),
                int(polewidth / 2),
                (255, 132, 100),
            )


        # flip coordinates
        self.surf = pygame.transform.flip(self.surf, False, True)

        p = precision = 3
        pos_left = 50
        u.text_to_screen(self.surf, f"phi1 {np.round(x[0], p)}", (pos_left, 50))
        u.text_to_screen(self.surf, f"phi2 {np.round(x[1], p)}", (pos_left, 70))
        u.text_to_screen(self.surf, f"ome1 {np.round(x[2], p)}", (pos_left, 90))
        u.text_to_screen(self.surf, f"ome2 {np.round(x[3], p)}", (pos_left, 110))
        if self.reward is not None:
            u.text_to_screen(self.surf, f"Rew {np.round(self.reward, p)}", (pos_left, 140))
        if self.action is not None:
            u.text_to_screen(self.surf, f"Act {np.round(self.action, p)}", (pos_left, 160))

        u.text_to_screen(self.surf, f"Episode {self.episode_count}", (43, 10))
        u.text_to_screen(self.surf, f"Step {self.ep_step_count}", (40, 30))


        self.screen.blit(self.surf, (0, 0))

        # reset button
        if self.reset_button is None:

            def req_res():
                self.request_reset = True

            self.reset_button = u.Button(
                self.screen, self.screen_width - 60, 5, 50, 20, u.red, u.light_red, "Reset", action=req_res
            )
        self.reset_button.show()

        if self.render_mode == "human":
            pygame.event.pump()

            # some event handling for interactivity
            push_angle = 10.0 / 180 * np.pi
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    pygame.quit()
                if ev.type == pygame.KEYDOWN:
                    # push rod to make cart do something
                    if ev.key == pygame.K_LEFT:
                        state = list(self.state)
                        state[2] -= push_angle
                        self.state = state
                    if ev.key == pygame.K_RIGHT:
                        state = list(self.state)
                        state[2] += push_angle
                        self.state = state

            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        # elif self.render_mode == "rgb_array":
        #     return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        state=None,
    ):
        super().reset(seed=seed)

        self.ep_step_count = 0
        self.episode_count += 1
        if state is None:
            self.state = np.zeros(self.observation_space.shape)
        else:
            self.state = state

        r = np.random.rand()*1.8
        phi = np.random.rand()*np.pi*2
        self.target = np.array([r*np.cos(phi), r*np.sin(phi)])

        if self.render_mode == "human":
            self.render()

        self.request_reset = False
        s = self.state

        return np.array(s, dtype=np.float32), {}

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False

