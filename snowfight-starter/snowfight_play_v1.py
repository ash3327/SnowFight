import random
from itertools import filterfalse

import __init__

import gym
import gym_snowfight  # there may be problems during import. check __init__.py in this directory for more.
from gym_snowfight.wrappers import RelativePosition, Compactify, CompactifyMore
import pygame
from cmdargs import args
from output import Output

"""
Running after doing 
        pip install -e gym-snowfight
Human player: run
        python snowfight-starter/snowfight_play_v1.py -m 'human'        # default fps = 20
Random player (command line interface): run
        python snowfight-starter/snowfight_play_v1.py
Random player (with GUI): run
        python snowfight-starter/snowfight_play_v1.py -m 'human_rand'   # default fps = 20
Plotting results:
        refer to output.py in the same directory for more information.
        
You are NOT advised to hold down more than 2 keys at once in the game.
If you hold down more than 1 key at once, the action taken in that step will be randomized to one of the keys.
Only one action can be taken per step.
"""

render_mode = args.mode
if render_mode == "human_rand":
    render_mode = "human"

episodes = args.episodes
max_steps = args.max_steps
window_size = args.window_size

env = gym.make('gym_snowfight/SnowFight', render_mode=render_mode, size=args.size,
               window_size=window_size, max_enemies=args.enemies)
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
env = Compactify(env)  #Compactify(env)

if args.fps is not None:
    env.metadata["render_fps"] = args.fps

env.action_space.seed(args.seed)
observation, info = env.reset(seed=args.seed)

output = Output(args.output_file, 'results', output_every_n=(1 if render_mode == 'human' else 100))  # does any output job.

episode = 0
total_score = 0
max_score = 0
high_tile = 0
total_steps = 0

running = True
step = 0

key_to_action = {
    pygame.K_LEFT: 0, pygame.K_RIGHT: 1,
    pygame.K_UP: 2, pygame.K_DOWN: 3, pygame.K_SPACE: 4
}
last_keys = []

while running and episode < episodes:
    action = None
    if args.mode == 'human':
        events = pygame.event.get()
        keys = pygame.key.get_pressed()
        pressed = False
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action = key_to_action[event.key]
                    if event.key not in last_keys:
                        last_keys += [event.key]
                    pressed = True
            elif event.type == pygame.QUIT:
                running = False
        if not pressed:
            last_keys = list(filterfalse(lambda x: not keys[x], last_keys))
            if len(last_keys) != 0:   # the last pressed keys are still pressed
                action = key_to_action[random.choice(last_keys)]
        if keys[pygame.K_s]:
            env.metadata["perspective_warp"] = min(1, env.metadata["perspective_warp"] + .1)
        elif keys[pygame.K_w]:
            env.metadata["perspective_warp"] = max(.1, env.metadata["perspective_warp"] - .1)
    else:
        action = env.action_space.sample()  # random
        if args.mode == 'human_rand':
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False

    observation, reward, done, truncated, info = env.step(action)
    # print(episode, action, observation, reward, info)
    step += 1

    if done or truncated or not running:
        output.log(done, episode, step, info)
        if done or not running:
            total_score += info['score']
            max_score = max(max_score, info['score'])

            total_steps += step

        observation, info = env.reset(seed=args.seed)
        episode += 1
        step = 0

if episode > 0:
    env.close()
    out = f"Average score: {total_score / episode:.2f}\n" \
          f"Maximum score: {max_score:.2f}\n" \
          f"Average steps: {total_steps / episode:.2f}"
    print(out)
    output.logs(out)
    output.output_img(episode=-1)


