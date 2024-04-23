import math
import random
from copy import copy

from .camera import *


class ObservableBlob(Blob, ABC):
    """ is the brain of each blob. display is done in the camera.py / BlobDisp class. """

    observations = {"x": -1., "y": -1., "facing": 2., "health": 3., "moving_dir": 0., "speed": 1.}

    def __init__(self, x=0, y=0, health=3, facing=0, moving_dir=0, speed=1,
                 texture: BlobTexture = copy(Blob.texture)):
        """ texture: a list of image file paths following the following pattern:"""
        if facing is None:
            facing = moving_dir
        super().__init__(x, y, facing=facing, texture=texture)
        self.observations.update({
           "moving_dir": moving_dir, "health": health,  "speed": speed
        })
        self.maxspeed = 2
        self.minspeed = -.5
        self.rot_amount = .03
        self.friction = .8
        self.acceleration = .2

    def update(self, action: int, direction: int, person='evil', snowballs: list = (), enemies: list = (), **kwargs) -> tuple[float, float, bool]:
        """
        Receives input as an action pair (act, dir),
        where act is either turning (0), changing speed (1) or shoot (2),
        and   dir is either negative (-1) or positive (+1).

        Please unpack the tuple before passing into this function.

        person is used to mark the source of the snowball, either 'evil' or 'player'.
        player's snowball won't harm himself.
        """
        self.observations["speed"] *= 1 - self.friction

        if action == 0:
            # changing moving direction, i.e. making a turn, would also change the gun direction.
            self.observations["moving_dir"] = (self.observations[
                                                   "moving_dir"] + self.rot_amount * direction + 1) % 2 - 1
            self.observations["facing"] = (self.observations["facing"] + self.rot_amount * direction + 1) % 2 - 1
        elif action == 1:
            self.observations["speed"] = \
                min(max(self.observations["speed"] + self.acceleration * direction, self.minspeed), self.maxspeed)

        speed, ang = self.observations["speed"], self.observations["moving_dir"]
        x, y = super()._update(speed, ang)

        rwd = 0
        scr = 0

        if action == 2 and direction == +1:  # i.e. throw snowball
            if len(snowballs) < Blob.max_snowballs:
                snowball = Snowball(x, y, z=.5, facing=self.observations["facing"], owner=person)
                snowballs.append(snowball)

                s_ang = self.to_polar(self.to_obs()[:3], snowball.to_obs()[:3])
                s_aimed_li = [self.to_polar(self.to_obs()[:3], e.to_obs()[:3])
                              for e in enemies]
                # give immediate reward when the agent shoots in the correct direction
                if any([
                    abs(s_aimed_li[i][1]-s_ang[2]) < self.rot_amount
                        and abs(s_aimed_li[i][0] < Snowball.range)
                        for i, e in enumerate(enemies)
                ]):
                    rwd += 1

        for snowball in snowballs:
            if (person == 'player') == snowball.observations["owner"]:
                # we hitting ourselves / enemy hitting themselves don't count.
                # we don't even check if they collide if it is the case.
                continue
            if self.collidewith(snowball):
                snowballs.remove(snowball)
                self.observations["health"] -= 1
                if person == 'evil' and snowball.observations["owner"] == 1:    # enemy is hit by us
                    rwd += 1
                    scr += 1
                if person == 'player' and snowball.observations["owner"] == 0:  # we are hit by enemy
                    rwd -= 1
                    scr -= 1
        for enemy in enemies:
            if self.collidewith(enemy):
                self.observations["health"] -= 1
                if person == 'player':  # we are hit by enemy
                    rwd -= 1
                    scr -= 1

        return rwd, scr, self.observations["health"] <= 0


class Player(ObservableBlob):
    observations = {"x": 0., "y": 0., "facing": 0., "moving_dir": 0., "health": 3., "speed": 0.,
                    "ammo": 10., "strength": 1.}

    def __init__(self, x=0, y=0, health=3, facing=0, moving_dir=0, ammo=10, speed=0, strength=1):
        super().__init__(x, y, health, facing, moving_dir, speed, Player.def_texture(strength))
        self.rel_size = strength
        self.observations.update({
            "ammo": ammo, "strength": strength
        })
        if Camera.display:
            self.sprite.texture.outline_color = (0, 0, 128)

    def update(self, action: int, direction: int, person='player', snowballs: list = (),
               enemies: list = (), **kwargs) -> tuple[float, float, bool]:
        """
        Receives input as an action pair (act, dir),
        where act is either turning (0), changing speed (1) or shoot (2),
        and   dir is either negative (-1) or positive (+1).

        Please unpack the tuple before passing into this function.
        """
        out = super().update(action, direction, person='player', snowballs=snowballs, enemies=enemies)
        self.observations["ammo"] = len(snowballs)#*-1+ Blob.max_snowballs
        return out


class EvilBlob(ObservableBlob):
    observations = {"x": -1., "y": -1., "facing": 2., "moving_dir": 0, "health": 0,
                    "shooting": 0, "speed": 0, "size": 0, "type": 0}
    passive_moves = (
        (0, +1), (0, -1), *((1, +1),)*5,
    )
    canmove = True

    def __init__(self, x=0, y=0, health=3, facing=0, moving_dir=0, shooting=0, speed=1, size=1,
                 blob_type=0):
        super().__init__(x, y, health, facing, moving_dir, speed, EvilBlob.def_texture(size))
        self.rel_size = size
        self.observations.update({
            "shooting": shooting, "size": size, "type": blob_type
        })
        if Camera.display:
            self.sprite.texture.outline_color = (128, 0, 0)
        self.repel_margin = .1
        self.target_ang = None
        self.maxspeed = 3   # ordinary EvilBlobs are little bit faster than Player.
        self.minspeed = self.maxspeed * -.2
        self.rot_amount = .05
        self.acceleration = .05

    def ai_move(self, player: Blob, **kwargs):
        if self.dist(player) > Camera.board_size*2/3:
            # strategy: wander around
            A = self.observations["x"] < Camera.board_size * self.repel_margin
            B = self.observations["x"] > Camera.board_size * (1-self.repel_margin)
            C = self.observations["y"] < Camera.board_size * self.repel_margin
            D = self.observations["y"] > Camera.board_size * (1-self.repel_margin)
            if A:
                self.target_ang = .25 if D else .75 if C else .5
            elif B:
                self.target_ang = -.25 if D else -.75 if C else -.5
            elif C:
                self.target_ang = 1
            elif D:
                self.target_ang = 0
            else:
                self.target_ang = None
        else:
            # strategy: locate player
            self.target_ang = self.get_ang(player)
        return self.target_ang

    def update(self, player: Blob, snowballs: list, **kwargs) -> tuple[float, float, bool]:
        """
        Receives a list of snowballs, and helps the enemy do a decision in a non-ML way.
        When snowballs hit the enemy, we would reduce health of the blob.
        """
        if EvilBlob.canmove:
            moves = self.passive_moves
            self.target_ang = self.ai_move(player)

            if self.target_ang is not None:
                ang_diff = self.target_ang - self.observations["moving_dir"]  # max(abs) is 1.
                if abs(ang_diff) > 1:
                    ang_diff = 2-ang_diff

                if ang_diff > self.rot_amount*2:
                    moves = ((0, +1),) * (5+int(abs(ang_diff)*10)) + ((1, +1),) * (5-int(abs(ang_diff)))
                elif ang_diff < -self.rot_amount*2:
                    moves = ((0, -1),) * (5+int(abs(ang_diff)*10)) + ((1, +1),) * (5-int(abs(ang_diff)))
                else:
                    self.target_ang = None
                """
                if ang_diff > .1:
                    moves = ((1, +1),) * int(abs(ang_diff)*10) + ((2, +1),) * int((1-abs(ang_diff))*10)
                elif ang_diff < -.1:
                    moves = ((1, -1),) * int(abs(ang_diff)*10) + ((2, +1),) * int((1-abs(ang_diff))*10)"""
            action, direction = random.choice(moves)
        else:
            action, direction = 2, -1

        return super().update(action, direction, person='evil', snowballs=snowballs)

    @classmethod
    def away_from_player(cls, player: Player, rand: np.random.Generator, size: int):
        pos = rand.random(size=2) * size
        while all(abs(pos - player.get_pos()) < Camera.board_size*.33):
            pos = rand.random(size=2)*size
        return pos


class GuardBlob(EvilBlob):
    def __init__(self, x=0, y=0, health=3, facing=0, moving_dir=0, shooting=0, speed=1, size=1, guarding_pos=(0, 0), radius=10):
        super().__init__(x, y, health, facing, moving_dir, shooting, speed, size)
        self.guarding_pos = guarding_pos
        self.radius = radius

    def ai_move(self, player: Blob, **kwargs):
        if self.dist(self.guarding_pos) > self.radius:
            self.target_ang = self.get_ang(self.guarding_pos)
        elif self.dist(player) < Camera.board_size/3:
            # strategy: locate player
            self.target_ang = self.get_ang(player)
        return self.target_ang


class Snowball(Blob):
    observations = {"x": -1., "y": -1., "facing": 2., "owner": 0}
    speed = 1
    z_vel = .6
    z_accel = -.1
    range = - speed * 2 * z_vel / z_accel

    def __init__(self, x=0., y=0., z=1., facing=0., owner='player'):
        super().__init__(x, y, z=z, facing=facing, texture=Snowball.def_texture(.66))
        self.rel_size = .66
        self.observations.update({"owner": int(owner=='player')})

    def update(self) -> tuple[float, float, bool]:
        """
        Receives a list of snowballs, and helps the enemy do a decision in a non-ML way.
        When snowballs hit the enemy, we would reduce health of the blob.
        """
        super()._update(self.speed, self.observations["facing"])
        self.z_vel += self.z_accel
        self.z += self.z_vel
        if self.z < 0:
            return 0, 0, True
        return 0, 0, False



