import math

import gym
from gym.spaces import Box
import numpy as np
from copy import copy


class RelativePosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.utils.flatten_space(env.observation_space)

    @staticmethod
    def to_polar(frm: np.ndarray, to: list) -> tuple[float, float, float]:
        def shift_ang(val: float, by: float) -> float:
            return (val - by + 1) % 2 - 1
        if to == [-1., -1., 2.]:
            return 0., 0., 0.  # the default "NOT-EXIST" value for the DQN model.
        rel_pos = np.array(to[:2]) - frm[:2]
        r, theta = sum(rel_pos**2)**.5, math.atan2(rel_pos[0], -rel_pos[1])/math.pi
        return r, shift_ang(theta, frm[2]), shift_ang(to[2], frm[2])

    def observation(self, obs) -> np.ndarray:
        """
        :returns: a flattened observation array.

        The returned array is in the following format before flattening:

            **Player (P)**
                x,              y,            (absolute) facing,(absolute) moving direction,
                1.04769611e-01  2.52447215e+01  2.43410895e-01  2.43410895e-01
                health,         speed,          ammo,           strength
                1.00000000e+00 -3.16912650e-71  6.00000000e+00  1.00000000e+00

            **Enemies**

            e1
                dist from P,     angle frm P, (relative) facing,(absolute) moving direction,
                9.40484800e-01  4.87823248e-01 -6.18270614e-01 -3.74859719e-01
                health,         speed,          shoot,           size,
                3.00000000e+00  9.93333760e-03  0.00000000e+00   1.00000000e+00
                type
                0.00000000e+00
            e2
                4.53492985e+00  2.09588934e-01 -6.80752673e-01 -4.37341778e-01
                3.00000000e+00  2.00399386e-01  0.00000000e+00  1.00000000e+00
                0.00000000e+00
            e3
                2.71174460e+01  2.77530710e-01  3.17279275e-01  5.60690170e-01
                3.00000000e+00  2.00336000e-01  0.00000000e+00  1.00000000e+00
                0.00000000e+00
            e4
                8.45710770e+00  3.96601511e-01 -3.98271755e-01 -1.54860860e-01
                3.00000000e+00  2.48386565e-01  0.00000000e+00  1.00000000e+00
                0.00000000e+00
            e5
                5.97045200e+00  2.83673081e-01 -8.51374684e-01 -6.07963789e-01
                3.00000000e+00  2.49999483e-01  0.00000000e+00  1.00000000e+00
                0.00000000e+00

            **Snowballs**

            s1
                dist from P,   angle frm P,   (relative) facing,owner (1 = player)
                1.10000000e+01 -1.11022302e-16  0.00000000e+00  1.00000000e+00
            s2
                1.00000000e+01 -1.11022302e-16  0.00000000e+00  1.00000000e+00
            s3
                9.00000000e+00 -1.11022302e-16  0.00000000e+00  1.00000000e+00
            s4
                8.00000000e+00 -1.11022302e-16  0.00000000e+00  1.00000000e+00
            s5
                6.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00
            s6
                5.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00
            s7
                0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
            s8
                0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
            s9
                0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
            s10
                0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
        """
        li = list(obs["player"])
        player = np.array(li[:3])   # doesn't need to change the coordinate representation of player.
        for enemy_obs in obs["enemies"]:
            enemy_obs = list(enemy_obs)
            enemy_obs[:3] = self.to_polar(player, enemy_obs[:3])  # r, theta
            li += enemy_obs
        for snowball_obs in obs["snowballs"]:
            snowball_obs = list(snowball_obs)
            snowball_obs[:3] = self.to_polar(player, snowball_obs[:3])  # r, theta
            li += snowball_obs
        return np.array(li)  # flattening of observation space.
