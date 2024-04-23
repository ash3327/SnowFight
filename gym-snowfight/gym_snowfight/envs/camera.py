import math

import gym
from gym import spaces
import pygame
import numpy as np
from functools import reduce
import operator
from abc import ABC, abstractmethod
from copy import copy

from .observable import Observable


class BlobTexture:
    perspective_warp = 1
    stroke_width = 1
    outline_color = (0, 128, 0)
    """ is the ratio of the new displayed height of rect on screen with respect to a character's fatness.
        it is universal for all BlobTextures. """

    def __init__(self, texture_dir: str = None, num_frames=8, num_angles=8,
                 width: int = 50, tall: int = None, fatness: int = None):
        """ texture_dir: the directory of the texture files. can be None, then outline is displayed.
                        The naming of files must follow the documentation, i.e., inside <texture_dir>:
                        angle0_frame0, angle0_frame1, etc.
            num_frames: the number of frames in an animation
            num_angles: the number of angles that a character has.

            width:      the displayed width of the character, same as size of hitbox.
            tall:       the displayed height of the character. keeps aspect ratio if None.
            fatness:    when viewed from directly above, the height of the character's hitbox."""
        self.num_frames = num_frames
        self.num_angles = num_angles
        self.angle_ratio = 2 * np.pi / num_angles if num_angles is not None else np.pi
        self.texture_dir = texture_dir

        self.frame = -1  # invalid state
        self.angle = -1  # invalid state

        self.width = width
        self.tall = tall
        self.fatness = fatness if fatness is not None else width
        # when viewed from directly above, the height of the character's hitbox.
        # assumed that the character is a circle if the parameter is not provided.
        self.surf = None

    def next(self, angle=0.) -> pygame.Surface:
        """ gets the next frame of the animation. """
        angle = (angle / 2 * self.num_angles) % self.num_angles if self.num_angles is not None else angle
        if angle != self.angle:
            self.frame = 0
            self.angle = angle
        else:
            self.frame += 1
            self.frame %= self.num_frames
            return self.surf
        if self.texture_dir is None:
            if self.tall is None:
                self.tall = self.width
            surf = pygame.Surface((self.width, self.tall))
            pygame.draw.rect(surf, (250, 250, 250), pygame.Rect(0, 0, self.width, self.tall),
                             width=0)
            pygame.draw.rect(surf, self.outline_color, pygame.Rect(0, 0, self.width, self.tall),
                             width=self.stroke_width)
            surf.set_colorkey((0, 0, 0), pygame.RLEACCEL)
        else:
            path = f"{self.texture_dir}/angle{int(self.angle)}_frame{self.frame}.png"
            if self.num_angles is None:
                path = f"{self.texture_dir}/frame{self.frame}.png"
            surf = pygame.image.load(path).convert_alpha()
            if self.tall is None:
                self.tall = self.width * surf.get_height()/surf.get_width()
            surf = pygame.transform.scale(surf, (self.width, self.tall))
        # TODO changing the surface image!!
        self.surf = surf
        return surf

    def hitbox(self) -> pygame.Surface:
        """ gets the hitbox surface """
        hitbox_tall = self.hitbox_tall()
        surf = pygame.Surface((self.width, hitbox_tall))
        pygame.draw.ellipse(surf, self.outline_color, pygame.Rect(0, 0, self.width, hitbox_tall),
                            width=self.stroke_width)
        center = np.array((self.width/2, hitbox_tall/2))
        offset = np.array((np.sin(self.angle*self.angle_ratio), -np.cos(self.angle*self.angle_ratio)))
        pygame.draw.line(surf, self.outline_color, center, (center*(1+offset)), width=self.stroke_width)
        surf.set_colorkey((0, 0, 0), pygame.RLEACCEL)

        return surf

    def hitbox_tall(self) -> int:
        return self.fatness * self.perspective_warp


class Blob(Observable, ABC):
    """ deals with the positions of the blob. """
    observations = {"x": 0.0, "y": 0.0, "facing": 0.0}
    texture = BlobTexture(texture_dir=None, num_frames=1, num_angles=8, width=30, tall=40)
    def_texture = lambda n: BlobTexture(texture_dir=None, num_frames=1, num_angles=8, width=30, tall=40)
    sprite = None
    display = False
    rel_size = 1
    max_snowballs = 10

    def __init__(self, x=0., y=0., facing=0., z=0., texture: BlobTexture = copy(texture)):
        """ texture: a list of image file paths following the following pattern:"""
        self.observations = {
            "x": x, "y": y, "facing": facing
        }
        self.texture = texture
        self.z = z
        if Blob.display:
            self.sprite = BlobDisp(self)

    @classmethod
    def get_len(cls) -> int:
        return len(cls.observations)

    def to_obs(self) -> tuple:
        return tuple(self.observations.values())

    @classmethod
    def null_to_obs(cls) -> tuple:
        return tuple(cls.observations.values())

    @classmethod
    def list_to_obs(cls, items: list, maxi: int) -> tuple:
        out = ()
        for item in items:
            out += (cls.to_obs(item),)
        if len(items) < maxi:
            out += tuple((cls.null_to_obs() for _ in range(len(items), maxi)))
        return out

    @classmethod
    def get_spaces(cls) -> gym.Space:
        return spaces.Box(low=-1, high=np.inf, shape=(cls.get_len(),))

    @abstractmethod
    def update(self, *args) -> tuple[float, bool]:
        """
        Updates the blob. It may be the case that the blob take some action, or simply do nothing.
        Returns an integer reward and whether the blob is dead.
        """
        pass

    def _update(self, speed: float, ang: float):
        ang *= math.pi
        x = self.observations["x"] + speed * math.sin(ang)
        y = self.observations["y"] - speed * math.cos(ang)
        self.observations["x"] = x = max(min(x, Camera.board_size), 0)
        self.observations["y"] = y = max(min(y, Camera.board_size), 0)
        return x, y

    @classmethod
    def get_X(cls, obj):
        if isinstance(obj, cls):
            return obj.observations["x"]
        elif isinstance(obj, tuple):
            return obj[0]
        return 0

    @classmethod
    def get_Y(cls, obj):
        if isinstance(obj, cls):
            return obj.observations["y"]
        elif isinstance(obj, tuple):
            return obj[1]
        return 0

    def relative_pos(self, other) -> np.ndarray:
        return np.array((Blob.get_X(other)-Blob.get_X(self),
                         Blob.get_Y(other)-Blob.get_Y(self)))

    def get_ang(self, other) -> float:
        rel_pos = self.relative_pos(other)
        return math.atan2(rel_pos[0], -rel_pos[1]) / math.pi

    def dist(self, other) -> float:
        rel_pos = self.relative_pos(other)
        return (rel_pos[0]**2 + rel_pos[1]**2) ** .5

    def collidewith(self, other) -> bool:
        assert isinstance(other, Blob)
        # note very carefully that the sizes of Player and EvilBlob
        # rel_size = 1 means one blob is 1 grid wide.
        rel_pos = self.relative_pos(other)
        if all(abs(rel_pos) < sum((self.rel_size, other.rel_size))/2):
            return True
        return False

    def get_pos(self) -> np.ndarray:
        return np.array([self.observations["x"],self.observations["y"]])

    @staticmethod
    def to_polar(frm: list, to: list) -> tuple[float, float, float]:
        """ returns: distance, relative angle, relative facing angle"""
        def shift_ang(val: float, by: float) -> float:
            return (val - by + 1) % 2 - 1
        if to == [-1., -1., 2.]:
            return 99., 0., 0.  # the default "NOT-EXIST" value for the DQN model.
        rel_pos = np.array(to[:2]) - np.array(frm[:2])
        r, theta = sum(rel_pos**2)**.5, math.atan2(rel_pos[0], -rel_pos[1])/math.pi
        return r, shift_ang(theta, frm[2]), shift_ang(to[2], frm[2])


class Camera:
    perspective_warp = 1
    board_size = 50
    grid_center = np.array((25., 25.))
    grid_angle = 0
    screen_size = (512, 512)
    grid_size = 30  # in pixels. all grids are squares distorted by 3d transformation.
    blob = None
    display = False

    @staticmethod
    def rotate(x: float, y: float, by_ang: float) -> np.ndarray:
        c = math.cos(by_ang*math.pi)
        s = math.sin(by_ang*math.pi)
        return np.array([x*c - y*s, x*s + y*c])

    @classmethod
    def normalize(cls, x: float, y: float):
        pos = np.array([x, y]) - cls.grid_center  # now the center is mapped to (0, 0)
        # the ENTIRE scene is rotated so that the screen is showing what the player sees.
        pos = cls.rotate(*pos, -cls.grid_angle)
        return pos

    @classmethod
    def to_screen(cls, x: float, y: float, z: float = 0) -> tuple[float] | None:
        if not cls.display:
            return

        pos = cls.normalize(x, y)

        # perspective wrapping by simply squishing the screen onto lower x axis.
        pos[1] = 3 - (1 - pos[1]) * cls.perspective_warp
        pos = pos * cls.grid_size                        # scaling so that each grid is 60 px
        pos += np.array(cls.screen_size) / 2        # now the center is mapped the center of screen.
        # if any object has z value, move it upward on screen. perspective_wrap = 0 maximizes this wrapping.
        pos[1] -= z*cls.grid_size*(1-cls.perspective_warp)
        return tuple(pos)

    @classmethod
    def follow_player(cls, b: Blob = None):
        if not cls.display:
            return
        if b is not None:
            cls.blob = b
        cls.grid_center[:] = np.array((b.observations["x"], b.observations["y"]))
        cls.grid_angle = b.observations["facing"]


class BlobDisp(pygame.sprite.Sprite):
    blob = None

    def __init__(self, blob: Blob):
        super(BlobDisp, self).__init__()
        self.blob = blob
        center = Camera.to_screen(
                self.blob.observations["x"], self.blob.observations["y"]
            )
        self.texture = self.blob.texture
        self.surf_display = self.texture.next(angle=0)  # first frame of angle zero (i.e. directly North)
        self.rect_display = self.surf_display.get_rect(
            center=(center[0], center[1]+self.texture.hitbox_tall()/2-self.texture.tall/2)
        )
        if self.texture.texture_dir is None:
            self.surf = self.blob.texture.hitbox()
            self.rect = self.surf.get_rect(
                center=center
            )

    def update(self):
        center = Camera.to_screen(
            self.blob.observations["x"], self.blob.observations["y"]
        )
        center2 = Camera.to_screen(
            self.blob.observations["x"], self.blob.observations["y"], self.blob.z
        )
        self.surf_display = self.texture.next(angle=self.blob.observations["facing"]-Camera.grid_angle)
        if self.texture.texture_dir is None:
            self.surf = self.blob.texture.hitbox()
            self.rect.height = self.texture.hitbox_tall()
            self.rect.center = center
        self.rect_display.center = (center2[0], center2[1]+self.texture.hitbox_tall()/2-self.rect_display.height/2)

    def __del__(self):
        self.kill()


class YAwareGroup(pygame.sprite.Group):
    # this section of code is borrowed from StackOverflow: Credit: user: sloth on Mar 19, 2019
    # https://stackoverflow.com/questions/55233448/pygame-overlapping-sprites-draw-order-based-on-location
    # the code is adjusted to match our needs.
    @staticmethod
    def by_y(sprite: BlobDisp) -> float:
        return Camera.normalize(sprite.blob.observations["x"], sprite.blob.observations["y"])[1]

    def draw(self, surface: pygame.Surface):
        sprites = self.sprites()
        surface_blit = surface.blit
        for sprite in sorted(sprites, key=self.by_y):
            self.spritedict[sprite] = surface_blit(sprite.surf_display, sprite.rect_display)
            if sprite.texture.texture_dir is None:
                self.spritedict[sprite] = surface_blit(sprite.surf, sprite.rect)
        self.lostsprites = []

