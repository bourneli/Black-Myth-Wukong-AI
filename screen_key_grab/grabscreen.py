# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 12:14:29 2020

@author: analoganddigital   ( GitHub )
"""

import cv2
import numpy as np
import win32gui, win32ui, win32con, win32api

def grab_screen(region=None):

    hwin = win32gui.GetDesktopWindow()

    if region:
            left,top,x2,y2 = region
            width = x2 - left + 1
            height = y2 - top + 1
    else:
        width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)
        left = win32api.GetSystemMetrics(win32con.SM_XVIRTUALSCREEN)
        top = win32api.GetSystemMetrics(win32con.SM_YVIRTUALSCREEN)

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)
    
    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height,width,4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return img


def crop_screen(screen_np, region):
    # 转换索引
    x_start, y_start, x_end, y_end = region

    # 添加assert
    assert screen_np.ndim >= 2, "screen_np.shape  is %s" % str(screen_np.shape)
    
    assert 0 <= x_start
    assert x_start <= x_end
    assert x_end <= screen_np.shape[1]
    
    assert 0 <= y_start
    assert y_start <= y_end
    assert y_end <= screen_np.shape[0]

    return screen_np[y_start:y_end, x_start:x_end]


if __name__ == "__main__":
    # 截取整个屏幕
    screenshot = grab_screen()

    # 显示截取的图像
    cv2.imshow("Screenshot", screenshot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 截取指定区域 (left, top, right, bottom)
    region = (100, 100, 500, 500)
    screenshot_region = grab_screen(region)
    print(type(screenshot_region))

    # 显示截取的区域图像
    cv2.imshow("Screenshot Region", screenshot_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()