# import atari_py
# all_games = atari_py.list_games()
# for game in all_games:
#     print(game)


import gymnasium as gym
import ale_py
gym.register_envs(ale_py)

all_envs = gym.envs.registry.keys()
# atari_envs = [env_spec.id for env_spec in all_envs if 'atari' in env_spec.entry_point]

for atari_env in all_envs:
    print(atari_env)




# import gymnasium as gym
# import ale_py

# gym.register_envs(ale_py)

# env = gym.make('ALE/Breakout-v5')
# obs, info = env.reset()
# obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# env.close()