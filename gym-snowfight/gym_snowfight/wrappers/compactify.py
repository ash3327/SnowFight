import math

import gym
from gym.spaces import Box
import numpy as np
from copy import copy
from functools import reduce


class Compactify(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        num_enemies = len(env.observation_space.get('enemies'))
        num_snowballs = len(env.observation_space.get('snowballs'))
        self.n = 6+num_enemies*6+num_snowballs*3+1
        self.observation_space = gym.spaces.Box(low=-1, high=np.inf,
                                                shape=(self.n,))

    @staticmethod
    def to_polar(frm: np.ndarray, to: list) -> tuple[float, float, float]:
        def shift_ang(val: float, by: float) -> float:
            return (val - by + 1) % 2 - 1
        if to == [-1., -1., 2.]:
            return 99., 0., 0.  # the default "NOT-EXIST" value for the DQN model.
        rel_pos = np.array(to[:2]) - frm[:2]
        r, theta = sum(rel_pos**2)**.5, math.atan2(rel_pos[0], -rel_pos[1])/math.pi
        return r, shift_ang(theta, frm[2]), shift_ang(to[2], frm[2])

    def observation(self, obs) -> np.ndarray:
        """
        :returns: a flattened observation array.

        The returned array is in the following format before flattening:

            **Player (P)**
                x,              y,            (absolute) facing,
                1.04769611e-01  2.52447215e+01  2.43410895e-01
                health,         speed,          ammo,
                1.00000000e+00 -3.16912650e-71  6.00000000e+00

            **Enemies**

            e1
                dist from P,     angle frm P, (relative) facing,
                9.40484800e-01  4.87823248e-01 -6.18270614e-01
                health,         speed,
                3.00000000e+00  9.93333760e-03
                type
                0.00000000e+00
            e2
                4.53492985e+00  2.09588934e-01 -6.80752673e-01
                3.00000000e+00  2.00399386e-01
                0.00000000e+00
            e3
                2.71174460e+01  2.77530710e-01  3.17279275e-01
                3.00000000e+00  2.00336000e-01
                0.00000000e+00
            e4
                8.45710770e+00  3.96601511e-01 -3.98271755e-01
                3.00000000e+00  2.48386565e-01
                0.00000000e+00
            e5
                5.97045200e+00  2.83673081e-01 -8.51374684e-01
                3.00000000e+00  2.49999483e-01
                0.00000000e+00

            Snowballs:

            s1
            ... dist, theta, facing (relative)
        """
        p = obs["player"]
        li = (*p[:3], *p[4:7])
        player = np.array(li[:3])   # doesn't need to change the coordinate representation of player.
        enemy_li = []
        for enemy_obs in obs["enemies"]:
            enemy_obs = list(enemy_obs)
            enemy_obs[:3] = self.to_polar(player, enemy_obs[:3])  # r, theta
            enemy_li += [(*enemy_obs[:3], *enemy_obs[4:6], enemy_obs[8])]
        enemy_li = sorted(enemy_li, key=lambda x: x[0])
        li += tuple(reduce(lambda x,y: x+y, enemy_li)) # sorted
        for snowball_obs in obs["snowballs"]:
            snowball_obs = list(snowball_obs)
            snowball_obs[:3] = self.to_polar(player, snowball_obs[:3])  # r, theta
            li += (*snowball_obs[:3],)
        li += (sum([1 for s in obs["snowballs"] if s[-1] == 0]),)
        return np.array(li)  # flattening of observation space.
