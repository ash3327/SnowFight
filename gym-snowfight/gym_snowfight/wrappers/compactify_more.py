import math

import gym
from gym.spaces import Box
import numpy as np
from copy import copy
from functools import reduce


class CompactifyMore(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_enemies = len(env.observation_space.get('enemies'))
        self.num_snowballs = len(env.observation_space.get('snowballs'))
        self.n = 4+self.num_enemies*3+self.num_snowballs*3
        self.board_size = env.size
        self.observation_space = gym.spaces.Box(low=-1, high=np.inf,
                                                shape=(self.n,))

    def to_polar(self, frm: np.ndarray, to: list) -> tuple[float, float, float]:
        def shift_ang(val: float, by: float) -> float:
            return (val - by + 1) % 2 - 1
        if to == [-1., -1., 2.]:
            return -1., 0., 0.  # the default "NOT-EXIST" value for the DQN model.
        rel_pos = np.array(to[:2]) - frm[:2]
        r, theta = sum(rel_pos**2)**.5, math.atan2(rel_pos[0], -rel_pos[1])/math.pi
        return r/self.board_size, shift_ang(theta, frm[2]), shift_ang(to[2], frm[2])

    def observation(self, obs) -> np.ndarray:
        """
        :returns: a flattened observation array.

        The returned array is in the following format before flattening:

            **Player (P)**
                x,              y,            (absolute) facing,
                1.04769611e-01  2.52447215e+01  2.43410895e-01
                ammo,
                6.00000000e+00

            **Enemies**

            e1
                dist from P,     angle frm P, (relative) facing,
                9.40484800e-01  4.87823248e-01 -6.18270614e-01
            e2
                4.53492985e+00  2.09588934e-01 -6.80752673e-01
        """
        p = obs["player"]
        li = (*(np.array(p[:3])/self.board_size), p[6]/self.num_snowballs)      # standardize to the range [0, 1] for better training.
        player = np.array(li[:3])   # doesn't need to change the coordinate representation of player.
        enemy_li = []
        for enemy_obs in obs["enemies"]:
            enemy_obs = list(enemy_obs)
            enemy_obs[:3] = self.to_polar(player, enemy_obs[:3])  # r, theta
            enemy_li += [(*enemy_obs[:3],)]
        enemy_li = sorted(enemy_li, key=lambda x: x[0])
        li += tuple(reduce(lambda x, y: x+y, enemy_li[:self.num_enemies]))  # sorted
        for snowball_obs in obs["snowballs"]:
            snowball_obs = list(snowball_obs)
            snowball_obs[:3] = self.to_polar(player, snowball_obs[:3])  # r, theta
            li += (*snowball_obs[:3],)
        #print(li[:4])
        return np.array(li)  # flattening of observation space.
