import cv2
import numpy as np

def main():
    # 读取本地的JPG文件
    image_path = "./img/blood_bar/normal2-smaller.png"
    image = cv2.imread(image_path)
    assert image is not None, "无法读取图像文件 %s" % image_path
    cv2.imshow("Original Image", image)
    #cv2.waitKey(0)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray Image", image)
    #cv2.waitKey(0)

    image = cv2.Canny(image, 120, 140)
    cv2.imshow("Canny Edges", image)

    # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, 
    #                              kernel = np.ones((3,3), np.uint8))
    # cv2.imshow("Morphology Processed", image)

    cv2.waitKey(0)


    # 查找轮廓
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 计算血条长度（轮廓的宽度）
    if len(contours) > 0:
        # 假设血条轮廓是最大的轮廓
        all_blood = 0
        for blood_contour in contours:
            _, _, w, _ = cv2.boundingRect(blood_contour)
            print(f"Health: {w}")

        # blood_contour = max(contours, key=cv2.contourArea)
        # _, _, w, _ = cv2.boundingRect(blood_contour)
        # print(f"Health: {w}")


if __name__ == '__main__':
    main()

