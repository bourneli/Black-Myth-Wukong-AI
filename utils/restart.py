import time
import pyautogui
import logging

logger = logging.getLogger(__name__)


def restart(initial = False):
    if initial == False:
        logger.debug("死,开始新一轮")
        time.sleep(3)
        # 以下用风灵月影满血以增加训练效率
        pyautogui.keyDown('num2')
        pyautogui.keyDown('num2')
        pyautogui.keyDown('num2') 
        time.sleep(1)
        pyautogui.keyUp('num2') 
        # pass
    else :
        pass
  
if __name__ == "__main__":  
    restart()