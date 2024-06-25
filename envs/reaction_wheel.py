import math
from typing import Optional, Union
import os
import numpy as np
from scipy.integrate import solve_ivp
from ipydex import IPS

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled

import util as u


class ReactionWheelEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    # TODO!!
    ## Description

    Reaction Wheel Pendulum

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation            | Min                 | Max               |
    |-----|------------------------|---------------------|-------------------|
    | 0   | Pole Angle             | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 1   | Pole Angular Velocity  | -Inf                | Inf               |
    | 2   | Wheel Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can be take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 475 for v1.

    ## Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)

    ## Arguments

    ```python
    import gymnasium as gym
    gym.make('CartPole-v1')
    ```

    On reset, the `options` parameter allows the user to change the bounds used to determine
    the new random state.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        # meta info
        self.set_name()
        self.seed = None
        self.episode_count = 0
        self.ep_step_count = 0
        self.total_step_count = 0
        self.training = False
        self.history = {
            "step": [],
            "episode": [],
            "state": [],
            "action": [],
            "reward": [],
            "terminated": [],
            "truncated": [],
            "info": [],
        }

        # parameters
        self.gravity = 9.8      
        self.masspole = 0.02
        self.masswheel = 0.3
        self.dist_pole_com = 0.063 # dist to center of mass (pole is twice as long)
        self.dist_wheel_com = 2 * self.dist_pole_com
        self.j_pole = 47e-6
        self.j_wheel = 32e-6
        # auxiliary values
        self.J = self.masspole*self.dist_pole_com**2 + self.masswheel*self.dist_wheel_com**2 + self.j_pole + self.j_wheel
        self.m = (self.masspole*self.dist_pole_com + self.masswheel*self.dist_wheel_com) * self.gravity
        self.parameters = [
            self.gravity,
            self.masspole,
            self.masswheel,
            self.dist_pole_com,
            self.dist_wheel_com,
            self.j_pole,
            self.j_wheel,
            self.J,
            self.m
            ]
        
        self.force_mag = 10.0
        self.tau = 0.002  # seconds between state updates
        self.kinematics_integrator = "solve_ivp"  # "euler"

        # Angle at which to fail the episode
        self.phi_threshold_radians = 12 * 2 * math.pi / 360

        # environment
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.phi_threshold_radians * 2,  # phi
                np.finfo(np.float32).max,  # phidot
                np.finfo(np.float32).max,  # theta
                np.finfo(np.float32).max,  # thetadot
            ],
            dtype=np.float32,
        )

        self.action_space = None
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 800
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        self.action = None
        self.reward = None

        # UI variables
        self.request_reset = False
        self.reset_button = None  # initialize some objects, that have to be persistent and not be recreated each step
        self.debug_button = None  # initialize some objects, that have to be persistent and not be recreated each step

        self.steps_beyond_terminated = None

    def set_name(self):
        self.name = self.__class__.__name__

    def get_force(self, action):
        raise NotImplementedError("This method has to be overwritten by subclass")

    def calc_new_state(self, action):
        phi, phi_dot, theta, theta_dot = self.state
        force = self.get_force(action)

        # based on mathematical pendulum

        def rhs(t, state):
            x1, x2, x3, x4 = state # phi, phi_dot, theta_dot
            try:
                u1 = force[0]
            except IndexError:
                u1 = force
                
            dx1_dt = x2
            dx2_dt = self.m / (self.J - self.j_wheel) * np.sin(x1) - 1 / (self.J - self.j_wheel) * u1
            dx3_dt = x4
            dx4_dt = - self.m / (self.J - self.j_wheel) * np.sin(x1) + 1 / self.j_wheel / (self.J - self.j_wheel) * u1

            return [dx1_dt, dx2_dt, dx3_dt, dx4_dt]

        tt = np.linspace(0, self.tau, 2)
        xx0 = np.array(self.state).flatten()
        s = solve_ivp(rhs, (0, self.tau), xx0, t_eval=tt)

        phi, phi_dot, theta, theta_dot = s.y[:, -1].flatten()

        #! Beschränkung für q1dot
        if abs(theta_dot) > 500:
            theta_dot = np.sign(theta_dot) * 500

        state = (phi, phi_dot, theta, theta_dot)
        return state

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
        self.action = action
        self.reward, terminated, truncated, info = self.get_reward()

        if self.render_mode == "human":
            self.render()

        # manipulate state to make interactive env with mobile target position
        state = np.array(self.state, dtype=np.float32)
        if self.request_reset:
            truncated = True

        self.save_step_data(state, action, self.reward, terminated, truncated, info)

        if hasattr(self, "post_processing_state"):
            state = self.post_processing_state(state)

        return state, self.reward, terminated, truncated, info

    def save_step_data(self, state, action, reward, terminated, truncated, info):
        self.history["step"].append(self.total_step_count)
        self.history["episode"].append(self.episode_count)
        self.history["state"].append(state)
        self.history["action"].append(action)
        self.history["reward"].append(reward)
        self.history["terminated"].append(terminated)
        self.history["truncated"].append(truncated)
        self.history["info"].append(info)

    def get_reward(self):
        phi, phi_dot, theta, theta_dot = self.state

        truncated = False
        info = {}
        terminated = bool(
            phi < -self.phi_threshold_radians
            or phi > self.phi_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        return reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        state=None,
    ):
        if seed is None and self.seed is not None:
            seed = self.seed + self.episode_count
        super().reset(seed=seed)

        if state is None:
            # random state
            low, high = self.c.get_reset_bounds(self)
            # self.state = self.np_random.uniform(low=low, high=high, size=self.observation_space.shape)
            self.state = self.np_random.uniform(low=low, high=high, size=np.array(low).shape)
        else:
            # fixed state
            self.state = state

        self.steps_beyond_terminated = None
        self.request_reset = False

        self.ep_step_count = 0
        self.episode_count += 1

        if self.render_mode == "human":
            self.render()

        s = self.state
        if hasattr(self, "post_processing_state"):
            s = self.post_processing_state(s)

        return np.array(s, dtype=np.float32), {}

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

        world_width = 5.0
        scale = self.screen_width / world_width
        polewidth = 10.0
        # polelen = scale * (2 * self.dist_pole_com)
        polelen = 200
        wheeldiameter = 50.0 * 2
        # cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        table_hight = 410 # carty
        # l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        # axleoffset = cartheight / 4.0
        # cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        # carty = 100  # TOP OF CART
        # cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        # cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        # gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        # gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))


        # Pole
        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[0])
            coord = (coord[0] + self.screen_width / 2.0, coord[1] + table_hight)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        # Wheel
        # Wheel center
        center = tuple((np.array(pole_coords[1]) + np.array(pole_coords[2])) // 2)     
        
        fname = os.path.join(os.path.dirname(__file__), "assets/wheel2.png")
        img = pygame.image.load(fname)
        scale_img = pygame.transform.smoothscale(
                img,
                (wheeldiameter, wheeldiameter),
            )
        rot_image = pygame.transform.rotate(scale_img, x[2]/np.pi*180)
        # rot_image = u.blit_rotate(self.surf, scale_img, center, x[2]/np.pi*180)
        self.surf.blit(rot_image, tuple(center - np.array(rot_image.get_rect().center)))
        
        # wheel joint 
        gfxdraw.aacircle(
            self.surf,
            int(center[0]),
            int(center[1]),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(center[0]),
            int(center[1]),
            int(polewidth / 2),
            (129, 132, 203),
        )
        
        # arrow
        fname = os.path.join(os.path.dirname(__file__), "assets/clockwise.png")
        img = pygame.image.load(fname)
        if x[-1] is not None:
            # theta_dot = np.sign(x[-1]) * np.min((np.abs(x[-1]), 100)) / 10
            theta_dot = np.sign(x[-1]) * np.log(np.abs(x[-1]) + 1)
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(theta_dot) / 2, scale * np.abs(theta_dot) / 2),
            )
            is_flip = bool(theta_dot < 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            self.surf.blit(
                scale_img,
                (
                    self.screen_width // 2 - scale_img.get_rect().centerx,
                    self.screen_width // 2 - scale_img.get_rect().centery + table_hight,
                ),
            )
        
        # joint at table
        gfxdraw.aacircle(
            self.surf,
            int(self.screen_width / 2.0),
            int(table_hight),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(self.screen_width / 2.0),
            int(table_hight),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, table_hight, (0, 0, 0))


        # flip coordinates
        self.surf = pygame.transform.flip(self.surf, False, True)

        # show state on screen
        p = precision = 4
        u.text_to_screen(self.surf, f"phi      {np.round(x[0], p)}", (int(self.screen_width / 2), 10))
        u.text_to_screen(self.surf, f"phidot   {np.round(x[1], p)}", (int(self.screen_width / 2), 30))
        u.text_to_screen(self.surf, f"theta    {np.round(x[2], p)}", (int(self.screen_width / 2), 50))
        u.text_to_screen(self.surf, f"thetadot {np.round(x[3], p)}", (int(self.screen_width / 2), 70))
        if self.reward is not None:
            u.text_to_screen(self.surf, f"Rew {np.round(self.reward, p)}", (int(self.screen_width *4/5), 100))
        if self.action is not None:
            u.text_to_screen(self.surf, f"Act {np.round(self.action, p)}", (int(self.screen_width *4/5), 120))

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

        # pause / debug button
        if self.debug_button is None:

            def ips():
                self
                IPS()

            self.debug_button = u.Button(
                self.screen,
                self.screen_width - 60,
                35,
                50,
                20,
                u.blue,
                u.light_blue,
                "Pause",
                text_color=u.white,
                action=ips,
            )
        self.debug_button.show()

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
                        state[0] -= push_angle
                        self.state = state
                    if ev.key == pygame.K_RIGHT:
                        state = list(self.state)
                        state[0] += push_angle
                        self.state = state
                # if ev.type == pygame.MOUSEBUTTONDOWN:
                #     mouse_pos = pygame.mouse.get_pos()
                #     # print(mouse_pos)
                #     if mouse_pos[1] > 300:
                #         target_x = -(self.screen_width / 2 - mouse_pos[0]) / scale
                #         self.target_offset = target_x
                #         print("Target", target_x)

            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class DefaultReactionWheelEnv(ReactionWheelEnv):
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__(render_mode)
        import envs.parameter.DefaultReactionWheel as c

        self.action_space = spaces.Box(-c.max_action, c.max_action, (1,), float)

        self.c = c
        self.seed = c.START_SEED

    def get_force(self, action):
        return action

    def get_reward(self):
        return self.c.get_reward(self)
