import time

import gym
from gym import spaces
import pygame
import numpy as np
from .observation import *


UNIVERSAL_GRID_SIZE = 50


class SnowFightEnv(gym.Env):
    metadata = {"render_modes": ["ai", "human", "rgb_array"],
                "render_fps": 40, "perspective_warp": .5}

    def __init__(self, render_mode=None, size=50, window_size=16, max_enemies=5, max_snowballs=10, render=False):
        # game environment initialization
        self.max_enemies = max_enemies
        self.max_snowballs = max_snowballs

        # display
        self.window_scale_factor = window_size
        self.window_size = 512 * window_size / 16  # The size of the PyGame window
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.size = size
        self.episode = 0
        self.Render = render
        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct frame rate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        # for evaluation of model
        # for RL training
        self._reset_params()

        # - observation and action spaces for RL trainings
        """
        Observation is somehow complicated. We will flatten all of that down to a 1-d list.
        
        Every Blob (including Player and EvilBlobs (enemies)) has the following 8 observable traits:
        position, health, facing, moving_dir, shooting, speed, size
        [  d , θ ,   hp  ,  ang  ,    mvang  ,   sh   ,  sp  ,  sz ] (position reduced into polar form d,θ)
        or [ x, y,   hp  ,  ang  ,    mvang  ,   sh   ,  sp  ,  sz ] (position not reduced by Wrapper)
        For Player, the traits "shooting" and "size" are replaced by "ammo" and "strength".
        
        And each Snowball (from Player and enemies) has the following 3 traits:
         position,  moving_dir
        [ d , θ  ,     mvang   ]
        """
        self.observation = Observation(max_enemies, max_snowballs, size, rand=self.np_random,
                                       window_size=self.window_size)
        self.observation_space = self.observation.get_spaces()

        # - We have 6 actions, corresponding to
        # "left_turn", "right_turn", "speed_up", "slow_down", "shoot" and "idle" (doing nothing)
        self.action_space = spaces.Discrete(6)

        # - action maps
        """
        The following dictionary maps abstract actions from `self.action_space` to an action pair (act, dir), 
        where act is either turning body (0), changing speed (1) or shoot (2), 
        and   dir is either negative (-1) or positive (+1).
        """
        self._action_to_direction = {
            0: (0, -1),  # left,       moving turn, negative
            1: (0, +1),  # right,      moving turn, positive
            2: (1, +1),  # accelerate, speed, positive
            3: (1, -1),  # decelerate, speed, negative
            4: (2, +1),  # shooting,   shoot, positive (true)
            5: (2, -1)   # idle,       shoot, negative (false)
        }
        self.shoot_action = 4
        self.idle_action = 5

        # the render_mode 'ai' is for GUI display of a RL agent.
        # we treat 'human' mode as a simple way of calling the GUI.
        self.ai = render_mode == 'ai'
        if self.ai:
            self.render_mode = 'human'

        #self.observation.respawn = False
        #EvilBlob.canmove = False

    def _get_obs(self):
        return self.observation.to_obs()

    def _get_info(self):
        return {
            "score": self.score,
            "kills": self.kill_count,
            "damage_dealt": self.total_damage,
            "damage_received": self.damage_received,
            "survived": self.survived_time
        }

    def _reset_params(self):
        # for evaluation of model
        self.score = 0
        self.kill_count = 0
        self.total_damage = 0
        self.damage_received = 0
        self.survived_time = 0

        # for RL training
        self.action = -1
        self.reward = 0

    def _game_over_screen(self):
        # TODO This is just the old one.
        # The following line copies our drawings from `canvas` to the visible window
        pygame.mixer.music.pause()
        s = pygame.Surface((self.window_size, self.window_size))
        s.set_alpha(196)
        s.fill((64, 64, 64))
        pygame.display.update(self.window.blit(s, (0, 0)))
        self._print_text(self.window, f'Game Over!', (self.window_size / 2, self.window_size / 2 - 100),
                         color=(255, 255, 255))
        self._print_text(self.window, f'Your Score is {self.score:.2f}.',
                         (self.window_size / 2, self.window_size / 2), color=(255, 255, 255))
        self.window.blit(self.window, self.window.get_rect())
        pygame.event.pump()
        pygame.display.update()
        if not self.ai:
            time.sleep(1)

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
        pygame.mixer.music.unpause()

    def _set_display(self):
        if self.window is None and self.render_mode in ("human", "rgb_array"):
            pygame.mixer.init()
            pygame.init()
            pygame.display.init()

            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            self.font = pygame.font.SysFont('Garamond', 50 * self.window_scale_factor // 16)
            self.medium_font = pygame.font.SysFont('Garamond', 30 * self.window_scale_factor // 16)

            # sound sources: Pixabay
            self.observation.sounds = {
                # sound credit: SoundsForHim / throwing clothes on floor
                'snowball': pygame.mixer.Sound("res/sounds/snowball.ogg"),
                # sound credit: Universfield / Breeze of Blood
                'blood': pygame.mixer.Sound("res/sounds/blood.mp3"),
                # sound credit: NeoSpica / Rock Smash
                'hit': pygame.mixer.Sound("res/sounds/hit.ogg"),
                # sound credit: tonsil5 / Zombie Growl 3
                'zombie': pygame.mixer.Sound("res/sounds/zombie.mp3"),
                # sound credit: tonsil5 / Zombie Death 2
                'zombie_death': pygame.mixer.Sound("res/sounds/zdeath.mp3"),
            }
            # sound credit: burning-mir / terror ambience
            pygame.mixer.music.load("res/sounds/bgmus.mp3")
            pygame.mixer.music.play(loops=-1)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        # we need the following line to seed self.np_random
        super().reset(seed=seed)

        if self.render_mode == "human":
            # show game over screen if game is over.
            if self.episode != 0:
                self._game_over_screen()

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            # setting 3d display.
            Camera.display = Blob.display = True
            BlobTexture.perspective_warp = Camera.perspective_warp = self.metadata["perspective_warp"]
            Camera.board_size = self.size  # setting 3d display camera.
            Camera.screen_size = (self.window_size, self.window_size)
            Camera.grid_size = UNIVERSAL_GRID_SIZE * self.window_scale_factor / 16

            if self.render_mode == 'human' or self.Render:
                self._set_display()
                Player.def_texture = lambda n: BlobTexture(texture_dir="res/player", num_frames=1, num_angles=None,
                                                           width=Camera.grid_size * n * 1.75,
                                                           tall=Camera.grid_size * n * 1.75 * 2)
                EvilBlob.def_texture = lambda n: BlobTexture(texture_dir="res/zombie", num_frames=1, num_angles=8,
                                                             width=Camera.grid_size * (n ** .5) * 1.75,
                                                             tall=Camera.grid_size * 1.25 * (n ** .5) * 1.75)
                Snowball.def_texture = lambda n: BlobTexture(texture_dir="res/snowball", num_frames=1, num_angles=None,
                                                             width=Camera.grid_size * .66, tall=Camera.grid_size * .66,
                                                             fatness=Camera.grid_size * .66)

        # reset every parameter before starting a new game
        self._reset_params()

        # initialize gameboard.
        self.observation.init()

        # getting the observation and info
        observation = self._get_obs()
        info = self._get_info()

        # renders the new game state onto the screen.
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # prepare the action.
        self.action = action
        BlobTexture.perspective_warp = Camera.perspective_warp = self.metadata["perspective_warp"]

        # update gamestate by taking the action. if the human player does nothing, then we simply treat
        # it as an idle action.
        if action is None:
            action = self.idle_action

        # An episode is done iff the agent has reached the target
        self.reward, sc, terminated = self.observation.update(self._action_to_direction[action],
                                                          acceleration= .1 + .02*(self.score//50))
        #print(self.reward)
        self.score += sc
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        self.episode += 1

        if action == self.idle_action:
            self.reward -= .1          # discourages idle
        if action == self.shoot_action:
            self.reward += .1          # encourages shooting

        return observation, self.reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    # printing text
    def _debug_text(self, canvas: pygame.Surface, text: str, pos=(300, 300), color=(150, 100, 100)):
        self.__print_some_text(self.medium_font, canvas, text, pos, color)

    def _print_text(self, canvas: pygame.Surface, text: str, pos=(300, 300), color=(100, 100, 100),
                    font=None, orientation='center'):
        if font is None:
            font = self.font
        self.__print_some_text(font, canvas, text, pos, color, orientation=orientation)

    def __print_some_text(self, font, canvas: pygame.Surface, text: str, pos=(300, 300),
                          color=(100, 100, 100), orientation='center'):
        text_surf = font.render(text, False, color)
        text_rect = text_surf.get_rect()
        match orientation:
            case 'center':
                text_rect.center = pos
            case 'left':
                text_rect.left = pos
            case 'top-left':
                text_rect.topleft = pos
            case 'right':
                text_rect.right = pos
            case 'top-right':
                text_rect.topright = pos

        canvas.blit(text_surf, text_rect)

    # display
    def _render_frame(self):
        self._set_display()

        # Drawing the Blobs.
        canvas = pygame.Surface((self.window_size, self.window_size))
        self.observation.draw(canvas)
        self._print_text(canvas, f"Score: {self.score:.2f}", pos=(10, 10), color=(0, 0, 0),
                         font=self.medium_font, orientation='top-left')

        # showing the score.
        # self._print_text(canvas, f'Score: {self.score}', pos=(self.window_size/2, self.bar_size/2))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.mixer.quit()
            pygame.quit()
