import gym
from gym import spaces
import numpy as np
import cv2  # 用于处理图像
import pyautogui  # 用于截图等操作
import keyboard # 执行会和人类相同，pyautogui执行键盘较慢。
import time
from screen_key_grab.grabscreen import crop_screen
from paddleocr import PaddleOCR
from collections import deque

def execute_keyboard(key, delay):
    keyboard.press(key)
    time.sleep(delay)
    keyboard.release(key)

class BlackMythWukongEnv(gym.Env):
    def __init__(self, 
                 game_left_top_x=0, game_left_top_y=40,
                 right_bottom_x=1680, right_bottom_y=1090):
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
        self._game_region = (game_left_top_x, game_left_top_y, right_bottom_x, right_bottom_y) # 游戏画面区域
        self._game_width = right_bottom_x - game_left_top_x
        self._game_height = right_bottom_y - game_left_top_y
        self._malo_blood_window = (round(self._game_width*0.110), self._game_height(self._game_height*0.867), 
                                   round(self._game_width*0.265) , round(self._game_height*0.878)) 
        self._boss_blood_window = (round(self._game_width*0.357), self._game_height(self._game_height*0.810), 
                                   round(self._game_width*0.657), round(self._game_height* 0.820)) 
        self._boss_defeated_window = (round(self._game_width*0.455), self._game_height(self._game_height*0.461) , 
                                      round(self._game_width*0.548) , round(self._game_height*0.497)) 
        # 动作延迟，模拟人类操作，单位：秒
        self._action_delay = 0.08 
        # 目前没有gpu加速，如果有需要，可以配置本地环境
        self._ocr = PaddleOCR(lang='ch',use_angle_cls=False, use_gpu=False)  

        self._history = deque(maxlen=100) # 存储malo和boss状态，用于计算reward
        
    def render(self, mode='human', close=False):
        """
        渲染环境，可在控制台或图形界面中显示状态活日志
        """
        pass

    def reset(self):
        """
        重置环境状态，并返回初始状态。
        
        返回:
            state (int): 重置后的初始状态
        """
        self._move_to_boss()
        original_screenshot = self._capture_game_screen()
        current_state = self._covert_screenshot_to_state(original_screenshot)
        return current_state

    def step(self, action):
        """
        执行给定的动作，并返回新的状态、奖励、是否终止和其他信息。
        
        参数:
            action (int): 智能体选择的动作
        
        返回:
            state (int): 新的状态
            reward (float): 当前步的奖励
            done (bool): 表示是否达到终止条件
            info (dict): 额外的诊断信息
        """
        self._take_action(action) # 执行一个动作，并更新环境状态

        screen_image = self._capture_game_screen() # 获取新状态（截图）
        next_state = self._covert_screenshot_to_state(screen_image)

        malo_blood = self._calculate_malo_blood(screen_image)
        boss_blood = self._calculate_boss_blood(screen_image)
        self._history.append((malo_blood,boss_blood))

        reward = self._calculate_reward() # 计算奖励，例如根据图片中的信息判断奖励
        done = self._check_done(screen_image) # 检查是否完成  
        info = {}# 返回状态、奖励、是否结束以及额外信息，可能需要将额外信息放在info里面，后面用来回溯扣血等信息。
        
        return next_state, reward, done, info

    def _capture_game_screen(self): 
        # 如果截屏性能存在瓶颈，可以考虑win32 API来截取
        return np.array(pyautogui.screenshot(region=self._game_region))
        
    def _covert_screenshot_to_state(self, screenshot):
        assert len(self.observation_space.shape) >= 2, "Observation space shape must have at least two dimensions."
        resized_screenshot = cv2.resize(screenshot, self.observation_space.shape[0:2]) # 调整大小以符合observation_space的定义
        return resized_screenshot

    def _take_action(self, action):
        # 执行动作，根据action的值选择游戏操作，例如模拟按键
        action_map = {
             0:"W", 1:"A", 2:"S", 3:"D", # 方向
             4:"O", 5:"K", 6:"ctrl", 7:"space", 8:"J", 9:"M" # 行为
        }
        assert action in action_map, "action = %d is not supported" % action 
        execute_keyboard(action_map[action], self._action_delay)
        
    def _move_to_boss(self):
        # 等待加载和重生
        while True:
            time.sleep(2) # 控制节奏，不用很频繁
            original_screen = self._capture_game_screen() # 获取新状态（截图）
            malo_blood_rate = self._calculate_malo_blood(original_screen, percentage=True)
            if malo_blood_rate > 0.975: # 理论上是100%，但是CV提取存在误差，所以没有设置100%
                break

        execute_keyboard("L", self._action_delay) # 锁定boss
        execute_keyboard("W", delay=10 ) # 走到boss面前


    def _calculate_reward(self):
        lastest_state = self._history[-10:0] # 10个差不多有1s
        reward = 0
        if lastest_state is None: # 没有历史数据
            return reward 

        # 这个地方取队列有问题，后面需要修改
        newest_malo_blood, newest_boss_blood = lastest_state[0]
        oldest_malo_blood, oldest_boss_blood = lastest_state[-1]            

        if newest_malo_blood - oldest_malo_blood < 0: # malo掉血，减分
            reward -= 10
        if newest_boss_blood - oldest_boss_blood < 0: # boss掉血，加分
            reward += 3
        return reward

    def _check_done(self, original_screen):
        done = False
        malo_blood = self._calculate_malo_blood(original_screen, percentage=True)
        if malo_blood < 0.01: # 检测malo的血量
            done = True
        if self._has_won(original_screen):
            done = True

        return done
    
    def _has_won(self, origin_screen):
        boss_state_img = crop_screen(origin_screen, self._boss_defeated_window)
        result = self._ocr.ocr(boss_state_img, cls=False) 
        boss_defeated = False
        for line in result:
            for box in line:
                text = box[1][0]
                if "击败" in text:
                    boss_defeated = True
                    break
            if boss_defeated:
                break
        return boss_defeated

    def _calculate_malo_blood(self, original_screen, percentage=False):
        malo_blood_image = crop_screen(original_screen, self._malo_blood_window)
        return self._detect_health_bar(malo_blood_image, percentage=percentage)
    
    def _calculate_boss_blood(self, original_screen, percentage=False):
        bloss_blood_image = crop_screen(original_screen, self._boss_blood_window)
        return self._detect_health_bar(bloss_blood_image, percentage=percentage)

    # 利用血条梯度变化，检测血条含量
    def _detect_health_bar(self, image_zero, percentage = False):

        image = cv2.GaussianBlur(image_zero, (3, 3), 0) # 剔除毛刺
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 转换为灰度图
        
        # 计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度的绝对值
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0) # 合并梯度
        _, thresh = cv2.threshold(grad, 5, 255, cv2.THRESH_BINARY) # 二值化处理
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
        
        if not contours: # 不存在轮廓
            return 0

        # 假设血条是梯度变化最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        
        # 计算血条参数
        _, _, w, h = cv2.boundingRect(max_contour)
        
        # 计算血条含量
        blood_value = (w / image_zero.shape[1]) * 100 if percentage else w * h
        
        return blood_value   


if __name__ == '__main__':
    pass