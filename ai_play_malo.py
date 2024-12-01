import gymnasium as gym
import torch.nn as nn
import torchvision.models as models
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
import register_env
    
class ResNetFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(ResNetFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # 加载预训练的ResNet模型
        resnet = models.resnet18(pretrained=True)
        
        # 冻结ResNet的前几层，仅解冻最后一个block和fc层
        for name, param in resnet.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        
        # 保留特征提取层
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        # 定义输出层，将ResNet的输出展平成features_dim的向量
        self.linear = nn.Linear(resnet.fc.in_features, features_dim)
        
    def forward(self, observations):
        # 将输入图像传递给ResNet模型，得到特征
        x = self.resnet(observations)
        x = x.view(x.size(0), -1)
        return self.linear(x)

# 自定义DQN网络结构，增加更多层
class CustomDQNPolicy(ActorCriticCnnPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(
            *args,
            features_extractor_class=ResNetFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=[512, 256, 128],
            **kwargs
        )


# 运行主函数
if __name__ == "__main__":
    # 创建自定义环境实例
    env = gym.make("BlackMythWukong-v0")

    # 创建DQN模型，使用自定义网络结构
    model = DQN(CustomDQNPolicy, env, verbose=1)
    # 训练模型
    model.learn(total_timesteps=10000)
    # 保存模型
    model.save("./model/dqn_resnet_blackmyth_custom")