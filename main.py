from vision_detection import VisionDetector
from serial_communication import SerialCommunication
import time


class Robot:
    def __init__(self):
        self.target_drinks = ["yykx", "wz", "bs", "yld"]#["营养快线", "旺仔", "百事", "养乐多"]
        self.shelf_drinks = ["riosmt", "dpty", "cp", "jdb"]#["锐澳水蜜桃", "东鹏", "茶π柠檬红茶", "加多宝"]
        self.collected_items = []
        self.vision_detector = VisionDetector()
        self.serial_comm = SerialCommunication(port='COM13')
        
        # 定义第三层的放置位置
        self.shelf_positions = [
            (100, 100, 200),  # 第一个放置位置
            (200, 100, 200),  # 第二个放置位置
            (300, 100, 200),  # 第三个放置位置
            (400, 100, 200)   # 第四个放置位置
        ]
        
        # 定义置物筐位置
        self.basket_position = (500, 100, 150)  # 置物筐位置
    
    
    
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
        
        # 获取物品的3D坐标（这里需要根据实际情况转换）
        x, y, z = self.convert_to_3d_coordinates(position)
        
        # 移动到物品上方
        self.serial_comm.move_to_position(x, y, z + 50)
        time.sleep(1)
        # 下降到物品位置
        self.serial_comm.move_to_position(x, y, z)
        time.sleep(1)
        # 抓取物品
        self.serial_comm.grab_item()
        # 抬起物品
        self.serial_comm.move_to_position(x, y, z + 50)
        time.sleep(1)
        
    def move_and_place(self, position):
        """
        移动到指定位置并放置物品
        Args:
            position: 目标位置坐标 (x, y, z)
        """
        x, y, z = position
        # 移动到目标位置上方
        self.serial_comm.move_to_position(x, y, z + 50)
        time.sleep(1)
        # 下降到目标位置
        self.serial_comm.move_to_position(x, y, z)
        time.sleep(1)
        # 释放物品
        self.serial_comm.release_item()
        # 抬起机械臂
        self.serial_comm.move_to_position(x, y, z + 50)
        time.sleep(1)
        
    def convert_to_3d_coordinates(self, bbox):
        """
        将2D边界框坐标转换为3D坐标
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
        Returns:
            tuple: (x, y, z) 3D坐标
        """
        # 这里需要根据实际情况实现坐标转换
        # 示例实现，需要根据实际相机参数和机器人坐标系调整
        x1, y1, x2, y2 = bbox
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        z = 150  # 默认高度，需要根据实际情况调整
        return (x, y, z)
        
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
        """执行完整任务"""
        try:
            # 1. 观察样品
            # sample_item = self.observe_sample_item()
            # print(f"检测到的样品物品: {sample_item}")
            
            # 2. 收集第二层饮料
            self.collect_second_level_drinks()
            print(f"已收集的饮料: {self.collected_items}")
            
            # # 3. 处理第一层饮料
            # self.handle_first_level_drinks()
            
            # # 4. 收集样品物品
            # self.collect_sample_items(sample_item)
            # print(f"最终收集的物品: {self.collected_items}")
            
        except Exception as e:
            print(f"任务执行出错: {str(e)}")
        finally:
            # 关闭串口连接
            self.serial_comm.close()

def main():
    robot = Robot()
    robot.execute_task()

