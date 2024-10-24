import unittest
import cv2
import numpy as np

from env_wukong import Wukong

class TestWukong(unittest.TestCase):
    
    def setUp(self):
        self.env = Wukong(observation_w=175, observation_h=200, action_dim=4)

    def test_all_malo_blood_count(self):
        self.assertEqual(self.env._all_malo_blood_count(), 138*11) 
        
    def test_norml_blood(self):
        # 读取本地的JPG文件
        image_path = "./img/screenshot_20241020-14_29_09.jpg"
        image = cv2.imread(image_path)
        self.assertTrue(image is not None, "无法读取图像文件 %s" % image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 自己裁剪
        x_start, y_start, x_end, y_end = self.env.self_blood_window
        blood_image = image[y_start:y_end, x_start:round(x_end)]

        cv2.imshow("Screenshot", blood_image)
        cv2.waitKey(0)

        blood_count = self.env._count_pixels(blood_image,
                               lower_bound=(210, 210, 210), 
                               upper_bound=(220, 220, 220))
        print(blood_count)
        print(self.env._all_malo_blood_count())
        print(blood_count/self.env._all_malo_blood_count())

        # 转成np格式
        # screenshot_np = np.array(blood_image)
        # cv2.imshow("Screenshot", screenshot_np)
        # cv2.waitKey(0)
        # print(blood_image.shape)
        # #print(np.min(blood_image))
        # #print(np.max(blood_image))

        # blood_count = self.env._count_pixels(screenshot_np,
        #                        lower_bound=(200, 200, 200), 
        #                        upper_bound=(220, 220, 220))
        # print(blood_count)


        # 转成hsv空间 #为什么要转成hsv，rgb不可以吗？
        # self_blood_hsv_img = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2HSV)
        # cv2.imshow("Screenshot", self_blood_hsv_img)
        # cv2.waitKey(0)
        # self_blood_hsv_img = screenshot_np

        # my_blood = self.env._malo_normal_blood_count(self_blood_hsv_img)
        # print("normal blood rate %.4f" % self.env._malo_blood_rate(my_blood))

        # my_blood = self.env._malo_buring_blood_count(self_blood_hsv_img)
        # print("buring blood rate %.4f" % self.env._malo_blood_rate(my_blood))

        # my_blood = self.env._malo_recover_blood_count(self_blood_hsv_img)
        # print("recover blood rate %.4f" % self.env._malo_blood_rate(my_blood))

        # self.assertGreater(my_blood, 1000, "当前血量 %s 过小" % my_blood)
        # self.assertLess(my_blood, 1100, "当前血量 %s 过大" % my_blood)

    def test_BuringBlood(self):
        image = cv2.imread("./img/tests/normal_blood_case_1.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 检查图像数据类型
        print(image.dtype)  # 应该输出   uint8  
        print(image.shape)

        # 检查图像的最小值和最大值
        print("Min value:", image.min())
        print("Max value:", image.max())  # 应该在0到255之间

        pass

    def test_NormalBlood(self):
        pass

if __name__ == '__main__':
    unittest.main()