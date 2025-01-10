# Deep Q-Learning Shooter Game Project "SnowFight" 
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-8B9467?style=flat&logo=openai)](https://gymnasium.farama.org/index.html)
[![Reinforcement Learning](https://img.shields.io/badge/Reinforcement_Learning-00BFFF?style=flat)]()
[![Group Project](https://img.shields.io/badge/Group_Project-FF9900?style=flat)]()
[![Last Updated](https://img.shields.io/badge/Last_Updated-December_2022-green?style=flat)]()

## Overview
- Created a Gym environment of a simple third-person shooter game in Python
- Implemented a simple Deep-Q Network with PyTorch to train agents to master at the game (left image)
- Fine-tuned the hyperparameters of the agent, achieving average kill streak of 7 (right image, top) and lengthend the survival duration by 4 times (right image, bottom), which significantly better than the random baseline of 0.22 kills on average.
- Explored how deep-Q learning models handle a variable quantity of moving objects, i.e. the bullets and enemies, and relevant adjustments to the reward functions and representations of the observation space needed.

**Report:** [![Report](https://img.shields.io/badge/Report-4285F4?style=flat&logo=github&logoColor=white&link=https://github.com/ash3327/SnowFight/blob/master/project%20report%20-%20group%205.pdf)](project%20report%20-%20group%205.pdf)

<img src="https://github.com/ash3327/ash3327/assets/86100752/60f36fa1-d6fd-490b-b275-19bb1cbe9715" width=300 height=300/>
<img src="https://github.com/ash3327/ash3327/assets/86100752/9ac9a3e3-8e36-436c-bbd9-48b80c06e2d6" width=400/>

## Game Design
- Survive a zombie apocalypse by controlling a snowball-throwing character.
- Goal: Survive as long as possible while killing zombies.
- Player has limited vision range.
- Game ends when a zombie touches the player.
- Image: Human Gameplay; Art: Myself; AI training: Myself; Observation and Reward Design: Myself & Jerry; Game Code: Jerry & Myself
<image src="docs/snowfight/gameplay-human-1.gif" width=300 height=300 />

## Model
- Deep Q-Network (DQN) model for agent training.
- 3-layer feedforward neural network in TensorFlow.
- Uses experience replay with batch training.
- Epsilon-decay strategy for interaction.

## Results
- Tested multiple decay factors (gamma).
- Adjusted rewards to improve agent's learning on long-term dependencies.
- Achieving an average kill streak of 7 and quadrupling survival time, far surpassing the random baseline of 0.22 kills on average.
<image src="docs/snowfight/results-1.png" width=400 height=400 />

## Notable Behaviors (1)
- AI learns to control its orientation for precise shooting.
- AI refines orientation control to improve shooting accuracy.
<image src="https://github.com/ash3327/ash3327/assets/86100752/60f36fa1-d6fd-490b-b275-19bb1cbe9715" width=300 height=300 />

## Notable Behaviors (2)
- AI learns to evade zombies by retreating to the map corner.
- AI retreats to a corner for better firing coverage.
<image src="docs/snowfight/results-2.gif" width=300 height=300 />

## Notable Behaviors (3)
- AI adopts a spinning and frequent shooting strategy.
- AI spins and shoots frequently to maximize hits.
<image src="docs/snowfight/results-3.gif" width=300 height=300 />

## Future Directions
- Further training needed due to time constraints of this project.
- Interest in refining rewards and exploring new mechanics in near future.