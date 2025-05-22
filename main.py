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
    # time.sleep(1)
    
    # dir = 0
    # robot.MOVE(2, 1000, 200)
    # robot.execute_task1(dir)
    # robot.execute_task2(dir)
    # robot.execute_task3()
    
    robot.execute_task()

    detection_thread.join()

if __name__ == "__main__":
    main() 
