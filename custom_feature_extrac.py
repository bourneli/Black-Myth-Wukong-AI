import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import ale_py

gym.register_envs(ale_py)


# 创建并包装环境
def make_env(env_id, seed=0):
    def _init():
        env = gym.make(env_id, render_mode='rgb_array')
        env = AtariWrapper(env)
        # env.seed(seed)
        return env
    return _init

# 使用 DummyVecEnv 包装环境
env_id = "ALE/Breakout-v5"
env = DummyVecEnv([make_env(env_id)])

# 创建模型
model = PPO("CnnPolicy", env, verbose=1)

# 训练模型
model.learn(total_timesteps=100000)  # 你可以根据需要调整时间步数

# 保存模型
model.save("ppo_breakout")

# 加载模型
model = PPO.load("ppo_breakout")

# 测试训练好的模型
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

# 关闭环境
env.close()