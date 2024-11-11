import gym
import numpy as np
import torch as th
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.envs import DummyVecEnv
from torchvision.models import resnet18
from gym import spaces

from black_myth_wukong_env import BlackMythWukongEnv

# 自定义特征提取器，包含ResNet18并允许其权重随训练更新
class ResNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super(ResNetFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # 使用预训练的ResNet18并删除最后的分类层
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Identity()  # 删除最后一层，保留特征向量
        self.flatten = nn.Flatten()
    
    def forward(self, observations: th.Tensor) -> th.Tensor:
        # 直接通过ResNet前向传播得到特征
        features = self.resnet(observations)
        return self.flatten(features)

# 定义训练函数AIPlayMalo
def AIPlayMalo(env: gym.Env, total_timesteps: int = 100000, model_save_path: str = "dqn_black_myth_model"):
    # 定义DQN的policy参数，使用自定义的ResNet特征提取器
    policy_kwargs = dict(
        features_extractor_class=ResNetFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),  # ResNet输出特征维度
        net_arch=[256, 128, 64]  # DQN的全连接层结构
    )
    
    # 初始化DQN模型
    model = DQN(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=10000,
        batch_size=32,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        gamma=0.99
    )
    
    # 开始训练
    model.learn(total_timesteps=total_timesteps)
    
    # 保存模型到本地
    model.save(model_save_path)
    print(f"模型已保存到 {model_save_path}.zip")
    
    return model

# 主函数
def main():
    # 创建环境，使用DummyVecEnv包装环境以兼容Stable-Baselines3
    env = DummyVecEnv([lambda: BlackMythWukongEnv( 
        game_left_top_x=0, game_left_top_y=40,
        right_bottom_x=1680, right_bottom_y=1090)])
    
    # 调用AIPlayMalo进行训练，并指定模型保存路径
    print("开始训练...")
    model = AIPlayMalo(env, total_timesteps=100000, model_save_path="./models/")
    print("训练完成！%s" % model)


# 运行主函数
if __name__ == "__main__":
    main()