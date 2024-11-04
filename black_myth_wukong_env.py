import gym
from gym import spaces
import numpy as np
import cv2  # 用于处理图像
import pyautogui  # 用于截图等操作
import keyboard # 执行会和人类相同，pyautogui执行键盘较慢。
import time

class BlackMythWukongEnv(gym.Env):
    def __init__(self):
        super(BlackMythWukongEnv, self).__init__()
        
        ##############################
        # 父类特征，需要重写
        ##############################
        # 假设输入为3通道RGB图像，尺寸为shape定义，opencv截取的图像格式为BGR4,所以需要转换
        self.observation_space = spaces.Box(low=0, high=255, shape=(512, 512, 3), dtype=np.uint8)
        # 定义动作空间，方向：WASD，行为：加速，躲闪，跳跃，轻击，击打，回血
        self.action_space = spaces.Discrete(4+6)

        ##############################
        # 自定义特征
        ##############################
        # 游戏画面所在区域
        self._game_region = (0, 40, 1680, 1090)
        # 动作延迟，模拟人类操作，单位：秒
        self._action_delay = 0.08 
        # 用于存储游戏状态或其他环境变量
        self._current_state = None
        
    def reset(self):
        self._move_to_boss()
        _, self._current_state = self._capture_game_screen()
        return self._current_state

    def step(self, action):
        self._take_action(action) # 执行一个动作，并更新环境状态
        original_screen, self._current_state = self._capture_game_screen() # 获取新状态（截图）
        reward = self._calculate_reward(original_screen) # 计算奖励，例如根据图片中的信息判断奖励
        done = self._check_done(original_screen) # 检查是否完成  
        info = {}# 返回状态、奖励、是否结束以及额外信息，可能需要将额外信息放在info里面，后面用来回溯扣血等信息。
        return self._current_state, reward, done, info

    def _capture_game_screen(self):        
        screenshot = np.array(pyautogui.screenshot(region=self._game_region)) # 截图作为状态输入
        assert len(self.observation_space.shape) >= 2, "Observation space shape must have at least two dimensions."
        resized_screenshot = cv2.resize(screenshot, self.observation_space.shape[0:2]) # 调整大小以符合observation_space的定义
        return (screenshot, resized_screenshot)

    def _take_action(self, action):
        # 执行动作，根据action的值选择游戏操作，例如模拟按键
        action_map = {
             0:"W", 1:"A", 2:"S", 3:"D", # 方向
             4:"O", 5:"K", 6:"ctrl", 7:"space", 8:"J", 9:"M" # 行为
        }
        assert action in action_map, "action = %d is not supported" % action 
        execute_action(action_map[action],  delay=self._action_delay)
        
    def _move_to_boss(self):
        # 这里需要寻路，大圣残躯寻路最简单，广智有点难
        # 1.等待加载和重生
        # 2.寻路到boss附近   
        # 3.锁定boss
        pass

    def _calculate_reward(self, original_screent):
        # 分析状态（图片）并计算奖励，可以在这里添加图像分析逻辑
        reward = 0
        # 示例：简单返回一个固定奖励
        # 实际情况可能是检测敌人数量、主角生命值等
        return reward

    def _check_done(self, state):
        # 检查是否满足结束条件，例如主角死亡或任务完成
        done = False
        # 示例：简单返回False，表示永不结束
        return done
    
def execute_action(action_key, delay = 0.05):
    keyboard.press(action_key)
    time.sleep(delay)
    keyboard.release(action_key)