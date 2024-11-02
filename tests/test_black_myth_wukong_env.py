import unittest
import numpy as np
from black_myth_wukong_env import BlackMythWukongEnv

class TestBlackMythWukongEnv(unittest.TestCase):
    
    def setUp(self):
        # 初始化环境实例
        self.env = BlackMythWukongEnv()
    
    def tearDown(self):
        # 关闭环境
        self.env.close()

    def test_initialization(self):
        # 检查action_space和observation_space
        self.assertEqual(self.env.action_space.n, 5, "Action space should have 5 discrete actions.")
        self.assertEqual(self.env.observation_space.shape, (480, 640, 3), "Observation space shape should be (480, 640, 3).")
        self.assertEqual(self.env.observation_space.dtype, np.uint8, "Observation space dtype should be uint8.")

    def test_reset(self):
        # 测试reset方法的输出
        observation = self.env.reset()
        self.assertTrue(isinstance(observation, np.ndarray), "Reset should return a numpy array.")
        self.assertEqual(observation.shape, (480, 640, 3), "Observation shape should be (480, 640, 3) after reset.")
        self.assertEqual(observation.dtype, np.uint8, "Observation dtype should be uint8 after reset.")

    def test_step(self):
        # 测试step方法的输出
        self.env.reset()
        action = self.env.action_space.sample()
        observation, reward, done, info = self.env.step(action)
        
        # 检查observation的类型和形状
        self.assertTrue(isinstance(observation, np.ndarray), "Step should return a numpy array for observation.")
        self.assertEqual(observation.shape, (480, 640, 3), "Observation shape should be (480, 640, 3) after step.")
        
        # 检查reward的类型
        self.assertTrue(isinstance(reward, (int, float)), "Reward should be a numeric value.")
        
        # 检查done的类型
        self.assertTrue(isinstance(done, bool), "Done should be a boolean value.")
        
        # 检查info的类型
        self.assertTrue(isinstance(info, dict), "Info should be a dictionary.")

    def test_render(self):
        # 渲染功能测试，不会检查图像显示，但测试代码运行不报错
        try:
            self.env.render()
        except Exception as e:
            self.fail(f"Render method failed with exception: {e}")

    def test_close(self):
        # 测试close方法，确保它可以顺利执行
        try:
            self.env.close()
        except Exception as e:
            self.fail(f"Close method failed with exception: {e}")

# 运行单元测试
if __name__ == '__main__':
    unittest.main()