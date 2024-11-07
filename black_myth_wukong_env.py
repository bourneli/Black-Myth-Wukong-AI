import gym
from gym import spaces
import numpy as np
import cv2  # 用于处理图像
import pyautogui  # 用于截图等操作
import keyboard # 执行会和人类相同，pyautogui执行键盘较慢。
import time
from screen_key_grab.grabscreen import crop_screen

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
        self._game_region = (0, 40, 1680, 1090) # 游戏画面区域
        self._malo_blood_window = (181, 908, 319, 959) # 天命人血条，相对于游戏画面
        # 动作延迟，模拟人类操作，单位：秒
        self._action_delay = 0.08 
        
    def reset(self):
        self._move_to_boss()
        original_screenshot = self._capture_game_screen()
        current_state = self._covert_screenshot_to_state(original_screenshot)
        return current_state

    def step(self, action):
        self._take_action(action) # 执行一个动作，并更新环境状态
        original_screen = self._capture_game_screen() # 获取新状态（截图）
        reward = self._calculate_reward(original_screen) # 计算奖励，例如根据图片中的信息判断奖励
        done = self._check_done(original_screen) # 检查是否完成  
        current_state = self._covert_screenshot_to_state(original_screen)
        info = {}# 返回状态、奖励、是否结束以及额外信息，可能需要将额外信息放在info里面，后面用来回溯扣血等信息。
        return current_state, reward, done, info

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

    def _check_done(self, original_screen):
        done = False
        malo_blood = self._calculate_malo_blood(original_screen, percentage=True)
        if malo_blood < 0.01: # 检测malo的血量
            done = True
        if self._has_won(original_screen):    # 检测“得胜”字样
            done = True

        return done
    
    def _has_won(self, origin_screen):
        return False

    def _calculate_malo_blood(self, original_screen, percentage=False):
        malo_blood_image = crop_screen(original_screen, self._malo_blood_window)
        return self._detect_health_bar(malo_blood_image, percentage=percentage)

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
    
def execute_action(action_key, delay = 0.05):
    keyboard.press(action_key)
    time.sleep(delay)
    keyboard.release(action_key)


if __name__ == '__main__':
    from PIL import Image
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

    # 加载图片
    image_path = './img/win_screenshot/win2.jpg'
    image = Image.open(image_path)

    # 使用 OCR 识别图片中的文字
    text = pytesseract.image_to_string(image, lang='chi_tra')  # 使用繁体中文识别

    # 检查是否包含“得勝”字样
    if "得勝" in text:
        print("图片中包含'得勝'字样。")
    else:
        print("图片中不包含'得勝'字样。")