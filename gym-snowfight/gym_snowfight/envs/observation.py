import collections

import gym
import pygame.transform
from gym import spaces
import numpy as np
from .blobs import *
from .camera import *
from .observable import *
from functools import reduce


class Observation(Observable):
    max_enemies = 5
    max_snowballs = 10
    size = 50   # the game board x, y value is in the range [0, size)
    # stores the game state.

    def __init__(self, max_enemies: int, max_snowballs: int, size: int, rand: np.random.Generator, window_size: int):
        self.player = None
        self.enemies, self.snowballs = [], []
        self.size = size
        self.rand = rand
        Observation.max_enemies = max_enemies
        Observation.max_snowballs = max_snowballs
        Blob.max_snowballs = max_snowballs
        self.all_sprites = YAwareGroup()
        self.window_size = window_size
        self.sounds = {}

        self.actions = collections.deque(maxlen=20)
        self.respawn = True

    def init(self):
        num_small = 0
        self.actions.clear()
        # randomly placing the player and the enemies on the board.
        self.player = Player(*(self.rand.random(size=2)*(self.size*.66)+self.size*.166),
                             facing=None, moving_dir=self.rand.random()*2-1,
                             speed=0, strength=1)
        Camera.follow_player(self.player)  # has no effect if it is not display mode.
        self.enemies = [
            EvilBlob(*EvilBlob.away_from_player(self.player, self.rand, self.size),
                     facing=None, moving_dir=self.rand.random() * 2 - 1,
                     speed=0, size=1)
            for _ in range(self.max_enemies-num_small)
        ] + [EvilBlob(*EvilBlob.away_from_player(self.player, self.rand, self.size),
                      facing=None, moving_dir=self.rand.random() * 2 - 1,
                      speed=0, size=.5) for _ in range(num_small)]
        """
        + [GuardBlob(*pos, facing=None, moving_dir=self.rand.random() * 2 - 1,
                      speed=0, size=1, guarding_pos=pos, radius=5) for pos in
             ((0, 0), (0, self.size-1), (self.size-1, 0), (self.size-1, self.size-1))]"""
        self.snowballs = []

        if Camera.display:
            self.all_sprites = YAwareGroup()
            self.all_sprites.add(self.player.sprite)
            for enemy in self.enemies:
                self.all_sprites.add(enemy.sprite)
            for snowball in self.snowballs:
                self.all_sprites.add(snowball.sprite)

    def update(self, action: np.ndarray, acceleration: float = None) -> tuple[float, float, bool]:
        self.actions.extend(action)
        snowball_cnt = len(self.snowballs)
        reward, score, died = self.player.update(*action, enemies=self.enemies, snowballs=self.snowballs)
        if snowball_cnt != len(self.snowballs):
            self.play_sound("snowball")
            snowball_cnt = len(self.snowballs)
        if died:
            self.play_sound("blood")
            return reward, score, True
        for enemy in self.enemies:
            rwd, scr, died = enemy.update(self.player, self.snowballs)
            if acceleration is not None:
                enemy.acceleration = acceleration
            if died:
                self.enemies.remove(enemy)
                self.play_sound("zombie_death")
                if self.respawn:
                    self.enemies.append(
                        EvilBlob(*EvilBlob.away_from_player(self.player, self.rand, self.size),
                                 facing=None, moving_dir=self.rand.random() * 2 - 1,
                                 speed=0, size=1)
                    )
            reward += rwd
            score += scr
        for snowball in self.snowballs:
            _, _, died = snowball.update()
            if died:
                self.snowballs.remove(snowball)
        if snowball_cnt != len(self.snowballs):
            self.play_sound("hit")
        if self.rand.random(size=()) < .02 * len([e for e in self.enemies if self.player.dist(e) < 10]):
            self.play_sound("zombie")

        Camera.follow_player(self.player)  # has no effect if it is not display mode.
        if len(self.enemies) == 0:
            return reward+10, score+10, True
        return reward, score, False

    def draw(self, canvas: pygame.Surface):
        assert Camera.display

        # draw the background
        canvas.fill((255, 255, 255))  # white

        for i in range(Camera.board_size):
            for j in range(Camera.board_size):
                center = Camera.to_screen(i+.5, j+.5)
                if not (-Camera.grid_size < center[0] < Camera.screen_size[0]+Camera.grid_size and
                        -Camera.grid_size < center[1] < Camera.screen_size[1]+Camera.grid_size):
                    continue
                pygame.draw.lines(canvas, (150, 200, 200), True, (
                    Camera.to_screen(i, j), Camera.to_screen(i+1, j),
                    Camera.to_screen(i+1, j+1), Camera.to_screen(i, j+1)
                ))
        # draw the blobs
        self.all_sprites = YAwareGroup()
        self.all_sprites.add(self.player.sprite)
        for enemy in self.enemies:
            self.all_sprites.add(enemy.sprite)
        for snowball in self.snowballs:
            self.all_sprites.add(snowball.sprite)
        self.all_sprites.update()
        self.all_sprites.draw(canvas)
        pass

    @classmethod
    def get_len(cls) -> int:
        return Player.get_len() \
               + cls.max_enemies * EvilBlob.get_len() \
               + cls.max_snowballs * Snowball.get_len()

    def to_obs(self):
        """
        return self.player.to_obs() \
               + EvilBlob.list_to_obs(self.enemies, self.max_enemies) \
               + Snowball.list_to_obs(self.snowballs, self.max_snowballs)
               """
        return {
            'player': self.player.to_obs(),
            'enemies': EvilBlob.list_to_obs(self.enemies, self.max_enemies),
            'snowballs': Snowball.list_to_obs(self.snowballs, self.max_snowballs)
        }

    @classmethod
    def get_spaces(cls) -> gym.Space:
        return spaces.Dict({
            'player': Player.get_spaces(),
            'enemies': spaces.Tuple((
                EvilBlob.get_spaces() for _ in range(cls.max_enemies)
            )),
            'snowballs': spaces.Tuple((
                Snowball.get_spaces() for _ in range(cls.max_snowballs)
            ))
        })

    def play_sound(self, name: str):
        if name in self.sounds:
            try:
                pygame.mixer.find_channel(force=name not in ("snowball", "hit")).play(self.sounds[name])
            except AttributeError:
                pass

