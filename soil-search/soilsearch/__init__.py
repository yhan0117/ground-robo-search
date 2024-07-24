from gymnasium.envs.registration import register

register(
    id="soil_search/SoilSearchWorld-v0",
    entry_point="soil_search.envs:SoilSearchEnv",
    max_episode_steps=301,
)