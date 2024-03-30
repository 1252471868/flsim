from gym.envs.registration import register

register(
    id="FLMARL-v0",
    entry_point="gym_flmarl.envs:FLMARL_Env",
    max_episode_steps=200,
    reward_threshold=195.0,
)