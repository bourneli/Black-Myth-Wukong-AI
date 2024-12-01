# import gymnasium as gym

# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines3.common.env_util import make_vec_env
# from stable_baselines3.common.utils import set_random_seed

# def make_env(env_id: str, rank: int, seed: int = 0):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: the environment ID
#     :param num_env: the number of environments you wish to have in subprocesses
#     :param seed: the initial seed for RNG
#     :param rank: index of the subprocess
#     """
#     def _init():
#         env = gym.make(env_id, render_mode="human")
#         env.reset(seed=seed + rank)
#         return env
#     set_random_seed(seed)
#     return _init

# if __name__ == "__main__":
#     env_id = "CartPole-v1"
#     num_cpu = 4  # Number of processes to use
#     # Create the vectorized environment
#     vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

#     # Stable Baselines provides you with make_vec_env() helper
#     # which does exactly the previous steps for you.
#     # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
#     # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

#     model = PPO("MlpPolicy", vec_env, verbose=1)
#     model.learn(total_timesteps=25_000)

#     obs = vec_env.reset()
#     for _ in range(1000):
#         action, _states = model.predict(obs)
#         obs, rewards, dones, info = vec_env.step(action)
#         vec_env.render()







# import gymnasium as gym

# from stable_baselines3 import DQN

# env = gym.make("CartPole-v1", render_mode="human")

# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("dqn_cartpole")

# del model # remove to demonstrate saving and loading

# model = DQN.load("dqn_cartpole")

# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     if terminated or truncated:
#         obs, info = env.reset()



import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")