from gym.envs.registration import register

register(
    id="gym_snowfight/SnowFight",
    entry_point="gym_snowfight.envs:SnowFightEnv",
)
