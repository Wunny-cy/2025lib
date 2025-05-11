from vision_detection import VisionDetector
from serial_communication import SerialCommunication
import time

def hook_item(self):
        """
        执行勾取动作
        """
        print(1)
        self.SVO(1, 160, 1000)
        self.serial_comm.check_ok()
        self.SVO(1, 80, 1000)
        self.serial_comm.check_ok()

if __name__ == '__main__':
    hook_item()