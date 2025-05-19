from robot import Robot
import threading
import time
from vision_detection import VisionDetector

detector = None
def show():
    global detector
    detector = VisionDetector()
    detector.start_detection()

def main():
    global detector
    detection_thread = threading.Thread(target=show)
    detection_thread.start()
    while detector == None: #等待初始化完成
        time.sleep(0.1)    
    robot = Robot(detector)
    # robot.DTG(2,1)
    time.sleep(1)
    
    # robot.execute_task1()
    # robot.execute_task2()
    robot.execute_task3()
    # robot.execute_task()

    detection_thread.join()

if __name__ == "__main__":
    main() 
