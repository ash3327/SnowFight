
import __init__

import gym
import gym_snowfight  # there may be problems during import. check __init__.py in this directory for more.
from gym.utils import play
import pygame

# this version is just to play the game in human-controlled mode
"""
Running after doing 
        pip install -e gym-snowfight
Human player: run
        python snowfight-starter/snowfight_play_v2.py
Passing in any sort of arguments in the command line WILL NOT BE ACCEPTED.
"""

env = gym.make('gym_snowfight/SnowFight', render_mode="rgb_array", size=30, render=True)

mapping = {
    (pygame.K_LEFT,): 0,  (pygame.K_LEFT,  pygame.K_UP): 0, (pygame.K_LEFT,  pygame.K_DOWN): 0,
    (pygame.K_RIGHT,): 1, (pygame.K_RIGHT, pygame.K_UP): 1, (pygame.K_RIGHT, pygame.K_DOWN): 1,
    (pygame.K_UP,): 2, (pygame.K_DOWN,): 3,
    (pygame.K_SPACE,): 4, (pygame.K_SPACE, pygame.K_UP): 4, (pygame.K_SPACE, pygame.K_DOWN): 4,
    (pygame.K_SPACE, pygame.K_UP, pygame.K_LEFT): 4, (pygame.K_SPACE, pygame.K_DOWN, pygame.K_LEFT): 4,
    (pygame.K_SPACE, pygame.K_UP, pygame.K_RIGHT): 4, (pygame.K_SPACE, pygame.K_DOWN, pygame.K_RIGHT): 4,
    (pygame.K_SPACE, pygame.K_LEFT): 4, (pygame.K_SPACE, pygame.K_RIGHT): 4,
}

play.play(env, keys_to_action=mapping, noop=None, fps=20)
