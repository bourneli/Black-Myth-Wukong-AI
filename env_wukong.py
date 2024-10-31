import pyautogui
import cv2
import time
import matplotlib.pyplot as plt
import utils.directkeys as directkeys
import numpy as np
from screen_key_grab.grabscreen import grab_screen, crop_screen
from screen_key_grab.getkeys import key_check
from utils.restart import restart
import logging

logger = logging.getLogger(__name__)

class Wukong(object):
    def __init__(self, observation_w, observation_h, action_dim):
        super().__init__()

        self.observation_dim = observation_w * observation_h
        self.width = observation_w
        self.height = observation_h
        self.death_cnt = 0
        self.action_dim = action_dim
        
        self.obs_window = (0,0,1677,1087)

        self.ding_shen_window = (1458, 851, 1459, 852) # 定身技能图标
        self.boss_blood_window = (668, 888, 1018, 900) # boss血条
        self.endurance_window = (183, 979, 305, 985) # 天命人体力
        self.self_blood_window = (181, 948, 319, 959) # 天命人血条

        # self.self_stamina_window = (180, 979, 304, 986) # 这就是体力
        
        
        self.boss_blood = 0
        self.self_blood = 0
        self.self_stamina = 0
        self.stop = 0
        self.emergence_break = 0

    def dingshen_available(self, current_screen):
        ding_shen_img = crop_screen(current_screen, self.ding_shen_window)
        hsv_img = cv2.cvtColor(ding_shen_img, cv2.COLOR_BGR2HSV)
        hsv_value = hsv_img[0,0]
        return hsv_value[2] >= 130

    def boss_blood_count(self, current_screen):
        boss_blood_img = crop_screen(current_screen, self.boss_blood_window)
        boss_blood_hsv_img = cv2.cvtColor(boss_blood_img, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 180])
        upper_white = np.array([360, 30, 220])
        mask = cv2.inRange(boss_blood_hsv_img, lower_white, upper_white)
        white_pixel_count = cv2.countNonZero(mask)
        return white_pixel_count
    
    def malo_endurence_count(self, current_screen):
        self_endurance_img = crop_screen(current_screen, self.endurance_window)
        endurance_gray = cv2.cvtColor(self_endurance_img,cv2.COLOR_BGR2GRAY)

        blurred_img = cv2.GaussianBlur(endurance_gray, (3,3), 0)
        canny_edges = cv2.Canny(blurred_img, 10, 100)
        value = canny_edges.argmax(axis=-1)
        return np.max(value)

    def malo_blood_count(self, current_screen):
        malo_blood_image = crop_screen(current_screen, self.self_blood_window)
        return self._detect_health_bar(malo_blood_image, percentage=False)
    
    

        
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

    def self_stamina_count(self, self_stamina_hsv_img): # 气力
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([360, 30, 220])
        mask = cv2.inRange(self_stamina_hsv_img, lower_white, upper_white)
        white_pixel_count = cv2.countNonZero(mask)
        return white_pixel_count

    def take_action(self, action):
        if action == 0:  # j
            directkeys.light_attack()
        elif action == 1:  # m
            directkeys.left_dodge()
        elif action == 2:
            directkeys.sanlian()
        elif action == 3:
            directkeys.right_dodge()
        elif action == 4:
            directkeys.hard_attack()
        elif action == 5:
            directkeys.stay_still()
        elif action == 6:
            directkeys.ding_shen_gong_ji()
        elif action == 7:
            directkeys.kan_po()

    def get_reward(self, 
                   boss_blood, next_boss_blood, 
                   self_blood, next_self_blood,
                   self_stamina, next_self_stamina,
                   stop, emergence_break, action, boss_attack):
        logger.info( "Self Blood: %d，Boss Blood: %d"  % (next_self_blood, boss_blood))
        if next_self_blood < 100:     # self dead 用hsv识别则量值大约在400，用canny大约在40
            logger.info("快死了，当前血量：%s, 下一个血量：%s" % (self_blood,next_self_blood))
            reward = -6
            done = 1
            stop = 0
            emergence_break += 1
            
            # 用风灵月影增加训练效率
            pyautogui.keyDown('num2')
            pyautogui.keyDown('num2')
            pyautogui.keyDown('num2') 
            time.sleep(1)
            pyautogui.keyUp('num2') 
            return reward, done, stop, emergence_break

        reward = 0
        self_blood_reward = 0
        boss_blood_reward = 0
        self_stamina_reward = 0
        if next_self_blood - self_blood < -5:
            self_blood_reward = (next_self_blood - self_blood) // 10
            logging.info("掉血惩罚:%s" % self_blood_reward)
            time.sleep(0.05)
            # 防止连续取帧时一直计算掉血
        if next_boss_blood - boss_blood <= -18:
            boss_blood_reward = (boss_blood - next_boss_blood) // 5
            boss_blood_reward = min(boss_blood_reward, 20)
            logging.info("打掉boss血而奖励:%s" % boss_blood_reward)

        if (action == 1 or action == 3) and boss_attack == True and next_self_stamina - self_stamina >= 7 and next_self_blood-self_blood == 0: # 存疑
            self_stamina_reward += 2
            logging.info("完美闪避奖励: %s" % self_stamina_reward)
        elif (action == 1 or action == 3) and boss_attack == True and next_self_blood-self_blood == 0:
            self_stamina_reward += 0.5
            logging.info("成功闪避奖励：%s" % self_stamina_reward)

        reward = reward + self_blood_reward * 0.8 + \
            boss_blood_reward * 1.2 + self_stamina_reward * 1.0
        logging.info("整体奖励：%s" % reward)
        done = 0
        emergence_break = 0
        return reward, done, stop, emergence_break

    def step(self, action, boss_attack):
        if (action == 0):
            logging.info("一连")
        elif (action == 1):
            logging.info("左闪避")
        elif (action == 2):
            logging.info("三连")
        elif action == 3:
            logging.info("右闪避")
        elif action == 4:
            logging.info("重棍")
        elif action == 5:
            logging.info("气力不足，歇脚一歇")
        elif action == 6:
            logging.info("定！五连绝世！")
        elif action == 7:
            logging.info("轻棍+识破")
        self.take_action(action)

        obs_screen = grab_screen(self.obs_window)

        

        # agent学习的输入
        obs_resize = cv2.resize(obs_screen, (self.width, self.height))
        obs = np.array(obs_resize).reshape(-1, self.height, self.width, 4)[0]

        # 状态统计，用于计算奖励
        next_self_blood = self.malo_blood_count(obs_screen)
        next_boss_blood = self.boss_blood_count(obs_screen)
        next_self_stamina = self.malo_endurence_count(obs_screen)
        reward, done, stop, emergence_break = self.get_reward(self.boss_blood, next_boss_blood, 
                                                              self.self_blood, next_self_blood,
                                                              self.self_stamina, next_self_stamina,
                                                              self.stop, self.emergence_break, 
                                                              action, boss_attack)
        self.self_blood = next_self_blood
        self.boss_blood = next_boss_blood
        self.self_stamina = next_self_stamina
        logging.info("当前自己的血量=%s，下一个自己的血量=%s, 当前boss血量=%s，下一个boss的血量=%s" % (self.self_blood, next_self_blood, self.boss_blood, next_boss_blood))
        return (obs, reward, done, stop, emergence_break)

    def pause_game(self, paused): # 用于训练中暂停
        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                logging.info('start game')
                time.sleep(1)
            else:
                paused = True
                logging.info('pause game')
                time.sleep(1)
        if paused:
            logging.info('paused--B2')
            while True:
                keys = key_check()
                if 'T' in keys:
                    if paused:
                        paused = False
                        logging.info('start game')
                        time.sleep(1)
                        break
                    else:
                        paused = True
                        time.sleep(1)
        return paused

    def reset(self, initial=False):
        restart(initial)
        obs_screen = grab_screen(self.obs_window)
        obs_resize = cv2.resize(obs_screen, (self.width, self.height))
        obs = np.array(obs_resize).reshape(-1, self.height, self.width, 4)[0]
        return obs


def sub_screen(region, screen_np):
    x_start, y_start, x_end, y_end = region
    # todo: 添加assert
    return screen_np[y_start:y_end, x_start:x_end]

def collect_screenshot():
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d-%H_%M_%S")
    image_file = './img/screenshot_%s.jpg' % ts

    env = Wukong(observation_w=175, observation_h=200, action_dim=4)
    screenshot = grab_screen(env.obs_window)

    

    # 显示截取的图像
    cv2.imshow("Screenshot", screenshot)
    cv2.imwrite(image_file, screenshot)
    print("Save image %s" % image_file)
    cv2.waitKey(0)

    region = (0, 100, 500, 1000)
    # x_start, y_start, x_end, y_end = region
    # screenshot = screenshot[y_start:y_end, x_start:x_end]

    screenshot = crop_screen(screenshot, region)

    cv2.imshow("Sub Screen", screenshot)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def extract_screenshot_info():
    env = Wukong(observation_w=175, observation_h=200, action_dim=4)
    

    # 读取本地的JPG文件
    #image_path = "./img/screenshot_20241020-14_32_19.jpg" # 正常血条1
    #image_path = "./img/screenshot_20241020-14_29_06.jpg" # 正常血条2 满血
    image_path = "./img/screenshot_20241020-14_33_09.jpg" # 被焚烧，闪烁血条  # 不够精确，需要单元测试来系统测试
    #image_path = "./img/screenshot_20241020-14_32_49.jpg" # 被焚烧，闪烁血条
    #image_path = "./img/screenshot_20241021-21_56_12.jpg" # 回血
    print("screen file: %s" % image_path)
    image = cv2.imread(image_path)

    # 检查图像是否成功读取
    assert image is not None, "无法读取图像文件 %s" % image_path
    print("图像形状:", image.shape)

    # 自己裁剪
    x_start, y_start, x_end, y_end = env.self_blood_window
    blood_image = image[y_start:y_end, x_start:x_end]

    screenshot_np = np.array(blood_image)
    cv2.imshow("Screenshot", screenshot_np)
    cv2.waitKey(0)

    self_blood_hsv_img = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2HSV)
    # cv2.imshow("Screenshot", self_blood_hsv_img)
    # cv2.waitKey(0)

    my_blood = env._malo_normal_blood_count(self_blood_hsv_img)
    print("正常血条判定B2 %s" % my_blood)

    my_blood = env._malo_buring_blood_count(self_blood_hsv_img)
    print("燃烧血条判定 %s" % my_blood)

    my_blood = env._malo_recover_blood_count(self_blood_hsv_img)
    print("回血血条判定 %s" % my_blood)

    my_blood = env._malo_blood_count(self_blood_hsv_img)
    print("综合血条判定 %s" % my_blood)

    # 在线获取图像HSV
    # https://pinetools.com/image-color-picker


if __name__ == '__main__':
    collect_screenshot()
    #extract_screenshot_info()
    