import cv2
import numpy as np

def main():
    # 读取本地的JPG文件
    image_path = "./img/blood_bar/red2.png"
    image = cv2.imread(image_path)
    assert image is not None, "无法读取图像文件 %s" % image_path
    cv2.imshow("Original Image", image)

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", gray)


    # 计算x方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # 计算y方向的梯度
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 计算梯度的绝对值
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # 合并梯度
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    # 显示结果
    cv2.imshow('Gradient', grad)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_health_bar(image_path):
    # 读取图像
    image_zero = cv2.imread(image_path)
    cv2.imshow('Image', image_zero)

    # 剔除毛刺
    #image = cv2.blur(image,(1,1))
    image = cv2.medianBlur(image_zero, 15)
    #image = cv2.GaussianBlur(image_zero, (1, 1), 0)
    cv2.imshow("Blur image", image)

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Gray', gray)
    # 计算x方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # 计算y方向的梯度
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # 计算梯度的绝对值
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    # 合并梯度
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    # 二值化处理
    _, thresh = cv2.threshold(grad, 5, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 假设血条是梯度变化最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 计算血条的长度
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # 计算血条含量
    health_percentage = (w / image.shape[1]) * 100
    
    # 在图像上绘制血条轮廓
    cv2.rectangle(image_zero, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('Health Bar', image_zero)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return health_percentage

 

def detect_health_bar_x(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算x方向的梯度
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    
    # 计算梯度的绝对值
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    
    # 二值化处理
    _, thresh = cv2.threshold(abs_grad_x, 5, 255, cv2.THRESH_BINARY)
    
    # 进行形态学操作以突出血条区域
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    eroded = cv2.erode(dilated, kernel, iterations=2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 假设血条是梯度变化最大的轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 计算血条的长度
    x, y, w, h = cv2.boundingRect(max_contour)
    
    # 计算血条含量
    health_percentage = (w / image.shape[1]) * 100
    
    # 在图像上绘制血条轮廓
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 显示结果
    cv2.imshow('Health Bar', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return health_percentage



if __name__ == '__main__':
    # main()

    # 使用示例
    #image_path = './img/blood_bar/normal3-smaller.png' # 有问题
    image_path = './img/blood_bar/normal4-smaller.png' 
    health_percentage = detect_health_bar(image_path)
    print(f'Health bar contains {health_percentage:.2f}%')


