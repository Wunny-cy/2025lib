from robot import Robot
import threading
import time
from vision_detection import VisionDetector

detector = None
def test():
    global detector
    detector = VisionDetector()
    detector.start_detection()


def main():
    global detector
    
    detection_thread = threading.Thread(target=test)
    # detection_thread.daemon = True
    detection_thread.start()
    while detector == None:
        time.sleep(1)    # robot.vision_detector.start_detection()
    robot = Robot(detector)
    robot.execute_task()

    detection_thread.join()

if __name__ == "__main__":
    main() 
