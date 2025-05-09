import cv2
import threading

class VideoStreamMonitor:
    def __init__(self):
        # 初始化代码...
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5000)  # 设置一个很大的值
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5000)

        cv2.namedWindow('Model Test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Model Test', 750, 1440)  # 设置初始窗口大小
        
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self.update)

    def start(self):
        self.thread.start()

    def update(self):
        while not self.stop_flag.is_set():
            ret, frame = self.cap.read()
            if ret:
                # 在这里进行你的图像处理操作
                # 逆时针旋转90度
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                height, width = frame.shape[:2]
                center_x = width // 2
                center_y = height // 2
                
                x1 = center_x - 120
                y1 = center_y + 180
                x2 = center_x + 120
                y2 = center_y - 200
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色矩形框
                
                # 计算九宫格分割线位置
                # 定义四条边线
                top_y = 0
                bottom_y = height
                left_x = 0
                right_x = width

                # 水平分割线
                middle_y1 = y1  # 目标框顶部y坐标作为第一分界线
                middle_y2 = y2  # 目标框底部y坐标作为第二分界线

                # 垂直分割线
                middle_x1 = x1  # 目标框左边x坐标作为第一分界线
                middle_x2 = x2  # 目标框右边x坐标作为第二分界线

                
                # 绘制水平分割线
                cv2.line(frame, (0, middle_y1), (width, middle_y1), (0, 255, 0), 1)
                cv2.line(frame, (0, middle_y2), (width, middle_y2), (0, 255, 0), 1)
                
                # 绘制垂直分割线
                cv2.line(frame, (middle_x1, 0), (middle_x1, height), (0, 255, 0), 1)
                cv2.line(frame, (middle_x2, 0), (middle_x2, height), (0, 255, 0), 1)
                # 显示处理后的图像
                cv2.imshow('Model Test', frame)

                # 等待1ms，以便窗口能够响应系统事件
                cv2.waitKey(1)
        self.cap.release()

    def stop(self):
        # 停止视频流处理线程
        self.stop_flag.set()
        self.thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = VideoStreamMonitor()
    monitor.start()

    try:
        input("按 Ctrl+C 停止监视...\n")
    except KeyboardInterrupt:
        print("停止中...")
    monitor.stop()
    print("已安全退出")