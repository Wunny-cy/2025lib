from vision_detection import VisionDetector
from serial_communication import SerialCommunication
import time


class Robot:
    def __init__(self):
        print("初始化")
        self.target_drinks = ["yykx", "wz", "bs", "yld"]#["营养快线", "旺仔", "百事", "养乐多"]
        self.shelf_drinks = ["riosmt", "dpty", "cp", "jdb"]#["锐澳水蜜桃", "东鹏", "茶π柠檬红茶", "加多宝"]
        self.collected_items = []
        self.vision_detector = VisionDetector()
        self.serial_comm = SerialCommunication(port='COM13',baudrate=115200)
        
        # 定义第三层的放置位置
        self.shelf_positions = [
            (100, 100, 200),  # 第一个放置位置
            (200, 100, 200),  # 第二个放置位置
            (300, 100, 200),  # 第三个放置位置
            (400, 100, 200)   # 第四个放置位置
        ]
        
        # 定义置物筐位置
        self.basket_position = (500, 100, 150)  # 置物筐位置
    
    def move_to_position(self, x, y, z):
        """
        控制机器人移动到指定位置
        Args:
            x: X坐标
            y: Y坐标
            z: Z坐标
        """
        command = f"G0 X{x} Y{y} Z{z}"
        self.serial_comm.send_command(command)
        # 等待移动完成
        while True:
            response = self.serial_comm.read_response()
            if response == "ok":
                break
            time.sleep(0.1)

    def grab_item(self):
        """
        执行抓取动作
        """
        self.serial_comm.send_command("M3")  # 打开夹爪
        time.sleep(1)
        self.serial_comm.send_command("M4")  # 关闭夹爪
        time.sleep(1)

    def release_item(self):
        """
        执行释放动作
        """
        self.serial_comm.send_command("M3")  # 打开夹爪
        time.sleep(1)
    
    def move_robot_to_center_item(self, position):
        """
        移动机器人使物品位于图像中心
        Args:
            position: 物品位置坐标 (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = position
        item_center_x = (x1 + x2) / 2
        image_center = self.vision_detector.get_image_center(self.vision_detector.get_camera_image())
        
        # 计算需要移动的距离
        move_distance = item_center_x - image_center[0]
        
        # 根据距离移动机器人
        if abs(move_distance) > 10:  # 设置一个阈值，避免抖动
            if move_distance > 0:
                self.serial_comm.send_command("G0 X10")  # 向右移动
            else:
                self.serial_comm.send_command("G0 X-10")  # 向左移动
            time.sleep(0.5)
    
    def move_and_grab(self, position):
        """
        移动到指定位置并抓取物品
        Args:
            position: 物品位置坐标 (x1, y1, x2, y2)
        """
        # 移动机器人使物品位于中心
        self.move_robot_to_center_item(position)
        
        # 获取物品的3D坐标
        x, y, z = self.vision_detector.convert_to_3d_coordinates(position)
        
        # 移动到物品上方
        self.move_to_position(x, y, z + 50)
        time.sleep(1)
        # 下降到物品位置
        self.move_to_position(x, y, z)
        time.sleep(1)
        # 抓取物品
        self.grab_item()
        # 抬起物品
        self.move_to_position(x, y, z + 50)
        time.sleep(1)
        
    def move_and_place(self, position):
        """
        移动到指定位置并放置物品
        Args:
            position: 目标位置坐标 (x, y, z)
        """
        x, y, z = position
        # 移动到目标位置上方
        self.move_to_position(x, y, z + 50)
        time.sleep(1)
        # 下降到目标位置
        self.move_to_position(x, y, z)
        time.sleep(1)
        # 释放物品
        self.release_item()
        # 抬起机械臂
        self.move_to_position(x, y, z + 50)
        time.sleep(1)
        
    def observe_sample_item(self):
        """观察高台小方桌上的样品物品"""
        # 获取图像
        image = self.vision_detector.get_camera_image()
        # 检测样品
        sample_item = self.vision_detector.detect_sample_item(image)
        if sample_item is None:
            raise Exception("未能检测到样品物品")
        return sample_item
        
    def collect_second_level_drinks(self):
        """从第二层收集指定饮料"""
        while len(self.collected_items) < len(self.target_drinks):
            # 获取图像
            image = self.vision_detector.get_camera_image()
            # 检测饮料
            detected_drinks = self.vision_detector.detect_drinks(image, self.target_drinks)
            
            for drink in detected_drinks:
                if drink['is_in_center'] and drink['name'] not in self.collected_items:
                    # 移动到饮料位置并抓取
                    self.move_and_grab(drink['position'])
                    # 移动到置物筐位置并放置
                    self.move_and_place(self.basket_position)
                    self.collected_items.append(drink['name'])
                    break
            
    def handle_first_level_drinks(self):
        """处理第一层饮料的上架任务"""
        placed_count = 0
        while placed_count < len(self.shelf_drinks):
            # 获取图像
            image = self.vision_detector.get_camera_image()
            # 检测饮料
            detected_drinks = self.vision_detector.detect_drinks(image, self.shelf_drinks)
            
            for drink in detected_drinks:
                if drink['is_in_center'] and placed_count < len(self.shelf_positions):
                    # 移动到饮料位置并抓取
                    self.move_and_grab(drink['position'])
                    # 移动到第三层指定位置并放置
                    self.move_and_place(self.shelf_positions[placed_count])
                    placed_count += 1
                    break
            
    def collect_sample_items(self, sample_item):
        """收集与样品相同的物品"""
        collected_count = 0
        while collected_count < 2:
            # 获取图像
            image = self.vision_detector.get_camera_image()
            # 检测与样品相同的物品
            detected_items = self.vision_detector.detect_drinks(image, [sample_item])
            
            for item in detected_items:
                if item['is_in_center'] and collected_count < 2:
                    # 移动到物品位置并抓取
                    self.move_and_grab(item['position'])
                    # 移动到置物筐位置并放置
                    self.move_and_place(self.basket_position)
                    self.collected_items.append(item['name'])
                    collected_count += 1
                    break
        
    def execute_task(self):
        """执行移动任务"""
        try:
            # 前进指定距离
            print("执行任务一：前进500单位")
            self.serial_comm.send_command("FWD")
            time.sleep(2)  # 等待移动完成
            
            # # 后退指定距离
            # print("执行任务二：后退500单位")
            # self.serial_comm.send_command("BWD")
            # time.sleep(2)
            
            # # 左移指定距离
            # print("执行任务三：左移500单位")
            # self.serial_comm.send_command("LFT")
            # time.sleep(2)
            
            # # 右移指定距离
            # print("执行任务四：右移500单位")
            # self.serial_comm.send_command("RGT")
            # time.sleep(2)
            
            # # 任务一
            # print("执行任务五：前进300单位")
            # self.serial_comm.send_command("TK1")
            # time.sleep(1.5)
            
            # # 任务二
            # print("执行任务六：后退300单位")
            # self.serial_comm.send_command("TK2")
            # time.sleep(1.5)
            
            # # 任务三
            # print("执行任务七：左移300单位")
            # self.serial_comm.send_command("TK3")
            # time.sleep(1.5)
            
            # # 任务四
            # print("执行任务八：右移300单位")
            # self.serial_comm.send_command("TK4")
            # time.sleep(1.5)
            
            # # 任务五
            # print("执行任务九：前进200单位")
            # self.serial_comm.send_command("TK5")
            # time.sleep(1)
            
            # # 任务六
            # print("执行任务十：后退200单位")
            # self.serial_comm.send_command("TK6")
            # time.sleep(1)
            
            # # 左转指定角度
            # print("执行任务十一：左转90度")
            # self.serial_comm.send_command("RTL")
            # time.sleep(1)
            
            # # 右转指定角度
            # print("执行任务十二：右转90度")
            # self.serial_comm.send_command("RTR")
            # time.sleep(1)
            
            print("所有移动任务执行完成")
            
        except Exception as e:
            print(f"任务执行出错: {str(e)}")
        finally:
            # 关闭串口连接
            self.serial_comm.close()

def main():
    robot = Robot()
    robot.execute_task()

if __name__ == "__main__":
    main() 
