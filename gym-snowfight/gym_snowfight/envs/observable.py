from abc import ABC, abstractmethod

import gym
from gym import spaces


class Observable(ABC):
    display = True

    @classmethod
    @abstractmethod
    def get_len(cls) -> int:
        """:returns: the length of the list returned by the function to_obs(). """
        pass

    @abstractmethod
    def to_obs(self):
        """:returns: flattening of all observable traits into a list. """
        pass

    @classmethod
    @abstractmethod
    def get_spaces(cls) -> gym.Space:
        """:returns: a gym.Space object"""
        pass
