from gymnasium.envs.registration import register

register(
    id='BlackMythWukong-v0',
    entry_point='black_myth_wukong_env:BlackMythWukongEnv', 
    max_episode_steps=1000,
)