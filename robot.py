from vision_detection import VisionDetector
from serial_communication import SerialCommunication
import time

class Robot:
    def __init__(self):
        print("初始化中。。。")
        self.target_drinks = ["yykx", "wz", "bs", "yld"]#["营养快线", "旺仔", "百事", "养乐多"]
        self.shelf_drinks = ["riosmt", "dpty", "cp", "jdb"]#["锐澳水蜜桃", "东鹏", "茶π柠檬红茶", "加多宝"]
        self.collected_items = []
        self.vision_detector = VisionDetector()
        self.serial_comm = SerialCommunication()
        self.DTG(1,1) #调
        self.SVO(1, 100, 1000) #调
        print("初始化完成")
        
        # 定义第三层的放置位置
        self.shelf_positions = [
            (100, 100, 200),  # 第一个放置位置
            (200, 100, 200),  # 第二个放置位置
            (300, 100, 200),  # 第三个放置位置
            (400, 100, 200)   # 第四个放置位置
        ]
        
        # 定义置物筐位置
        self.basket_position = (500, 100, 150)  # 置物筐位置
        
        # 定义标签槽位置
        self.label_positions = [
            (100, 50, 200),   # 第一个标签槽位置
            (200, 50, 200),   # 第二个标签槽位置
            (300, 50, 200),   # 第三个标签槽位置
            (400, 50, 200)    # 第四个标签槽位置
        ]
    
    def DTG(self, id, dir):
        """
        控制电推杆移动
        Args:
            id: 电推杆编号 (1-4)
            dir: 移动方向 (0:后退, 1:前进)
        """
        if id not in [1, 2, 3, 4] or dir not in [0, 1]:
            raise ValueError("无效的电推杆编号或方向")
        self.serial_comm.send_command(f"DTG {id} {dir}")  # 控制电推杆
        time.sleep(1)

    def SVO(self, id, angle, delay):
        """
        控制舵机转动
        Args:
            id: 舵机编号 (1-8)
            angle: 转动角度 (0-180度)
            delay: 延迟时间 (毫秒)
        """
        time.sleep(3)
        if id not in range(1, 9):
            raise ValueError("无效的舵机编号，必须在1-8之间")
        if not 0 <= angle <= 180:
            raise ValueError("无效的角度值，必须在0-180度之间")
        if delay < 0:
            raise ValueError("延迟时间不能为负数")
            
        self.serial_comm.send_command(f"SVO {id} {angle} {delay}")  # 控制舵机
        time.sleep(delay / 1000)  # 将毫秒转换为秒

    def ARM_grab_item(self):
        """
        执行抓取动作
        """
        self.serial_comm.send_command("M3")  # 打开夹爪
        time.sleep(1)
        self.serial_comm.send_command("M4")  # 关闭夹爪
        time.sleep(1)

    def ARM_release_item(self):
        """
        执行释放动作
        """
        self.serial_comm.send_command("M3")  # 打开夹爪
        time.sleep(1)

    def hook_item(self):
        """
        执行勾取动作
        """
        self.serial_comm.send_command("M4")  # 打开勾爪
        time.sleep(1)
        self.serial_comm.send_command("M4")  # 关闭勾爪
        time.sleep(1)

    def move(self, command):
        """
        控制机器人移动
        Args:
            command: 移动指令，格式为 "X<距离>" 或 "Y<距离>"
                    X轴：正值表示右移，负值表示左移
                    Y轴：正值表示前进，负值表示后退
        """
        # 检查指令格式
        if not (command.startswith('X') or command.startswith('Y')):
            raise ValueError("指令必须以 X 或 Y 开头")
        
        # 解析指令
        axis = command[0]  # X 或 Y
        try:
            distance = int(command[1:])
        except ValueError:
            raise ValueError("距离必须是整数")
            
        # 验证距离范围
        if not 1 <= abs(distance) <= 2000:
            raise ValueError("移动距离必须在1-2000毫米之间")
            
        # 根据轴和方向发送对应指令
        if axis == 'X':
            if distance > 0:
                # 右移
                self.serial_comm.send_command(f"RGT {distance}")
            else:
                # 左移
                self.serial_comm.send_command(f"LFT {abs(distance)}")
        else:  # axis == 'Y'
            if distance > 0:
                # 前进
                self.serial_comm.send_command(f"FWD {distance}")
            else:
                # 后退
                self.serial_comm.send_command(f"BWD {abs(distance)}")
                
        # 等待移动完成
        while True:
            response = self.serial_comm.read_response()
            if "ok" in response.lower():
                break
            time.sleep(0.1)

    def move_to_position(self, x, y):
        """
        控制机器人移动到指定位置
        Args:
            x: X坐标
            y: Y坐标
        """
        # 先移动X坐标
        self.move(f"X{x}")
        # 再移动Y坐标
        self.move(f"Y{y}")

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
        if abs(move_distance) > 30:  # 设置一个阈值，避免抖动
            if move_distance > 0:
                self.move("X30")  # 向右移动
            else:
                self.move("X-30")  # 向左移动
            time.sleep(0.5)
    
    def move_and_grab(self, position):
        """
        移动到指定位置并用机械臂抓取物品
        Args:
            position: 物品位置坐标 (x1, y1, x2, y2)
        """
        # 移动机器人使物品位于中心
        self.move_robot_to_center_item(position)
        
        # 获取物品的3D坐标
        x, y, z = self.vision_detector.convert_to_3d_coordinates(position)
        # 伸出夹爪
        self.serial_comm.send_command("M4")
        # 移动到物品上方
        self.move_to_position(x, y)
        time.sleep(1)
        # 下降到物品位置
        self.move_to_position(x, y)
        time.sleep(1)
        # 抓取物品
        self.ARM_grab_item()
        # 抬起物品
        self.move_to_position(x, y)
        # 收回夹爪
        self.serial_comm.send_command("M3")
        time.sleep(1)
        
    def move_and_place(self, position):
        """
        移动到指定位置并放置物品
        Args:
            position: 目标位置坐标 (x, y)
        """
        x, y, z = position
        # 伸出夹爪
        self.ARM_grab_item()
        # 移动到目标位置上方
        self.move_to_position(x, y)
        time.sleep(1)
        # 下降到目标位置
        self.move_to_position(x, y)
        time.sleep(1)
        # 释放物品
        self.ARM_release_item()
        # 抬起机械臂
        self.move_to_position(x, y)
        # 收回夹爪
        self.ARM_release_item()
        time.sleep(1)
        
    def move_and_hook(self, position):
        """
        移动到指定位置并勾取物品
        Args:
            position: 物品位置坐标 (x1, y1, x2, y2)
        """
        # 移动机器人使物品位于中心
        self.move_robot_to_center_item(position)
        time.sleep(1)
        # 勾取物品
        self.hook_item()
        time.sleep(1)

    def go_ahead(self):
        """
        控制机器人绕货架前进
        """
        # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
        command = f"FWD 100"
        self.serial_comm.send_command(command)

    def RT(self, angle):
        """
        控制机器人原地旋转
        Args:
            angle: 旋转角度 (-180.0 ~ 180.0度)
        """
        if not -180.0 <= angle <= 180.0:
            raise ValueError("无效的旋转角度，必须在-180.0到180.0度之间")
            
        self.serial_comm.send_command(f"RT {angle:.1f}")  # 控制旋转，保留一位小数
        # 等待旋转完成
        while True:
            response = self.serial_comm.read_response()
            if "ok" in response.lower():
                break
            time.sleep(0.1)
            
    def observe_sample_item(self):
        """观察高台小方桌上的样品物品"""
        self.move_to_position(100, 100)
        # 获取图像
        image = self.vision_detector.get_camera_image()
        # 检测样品
        sample_item = self.vision_detector.detect_sample(image)
        if sample_item is None:
            raise Exception("未能检测到样品物品")
        return sample_item
        
    def collect_sample_items(self):
        """收集与模板匹配的物品"""
        collected_count = 0
        while collected_count < 2:
            # 检测与模板匹配的物品
            result = self.vision_detector.detect_sample_item()
            
            if result and result['confidence'] > 0.5:  # 设置置信度阈值
                x_min, y_min, x_max, y_max = result['position']
                # 计算物品中心位置
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                
                # 检查物品是否在图像中心区域
                if abs(center_x - self.vision_detector.get_image_center()[0]) < self.vision_detector.image_center_threshold:
                    # 转换为3D坐标
                    position_3d = self.vision_detector.convert_to_3d_coordinates((x_min, y_min, x_max, y_max))
                    # 移动到物品位置并抓取
                    self.move_and_grab(position_3d)
                    # 移动到置物筐位置并放置
                    self.move_and_place(self.basket_position)
                    collected_count += 1
                    print(f"已收集第 {collected_count} 个物品")
            
            # 短暂延迟，避免过于频繁的检测
            time.sleep(0.1)
        
    def collect_second_level_drinks(self):
        """从第二层收集指定饮料"""
        while len(self.collected_items) < len(self.target_drinks):
            # 控制机器人绕货架前进
            print("前进")
            self.go_ahead()
            self.RT(90)
            # 获取图像
            image = self.vision_detector.get_camera_image()
            # 检测饮料
            detected_drinks = self.vision_detector.detect_drinks(image, self.target_drinks)
            
            for drink in detected_drinks:
                if drink['is_in_Xcenter'] and drink['name'] not in self.collected_items:
                    # 移动到饮料位置并勾取
                    self.move_and_hook(drink['position'])
                    self.collected_items.append(drink['name'])
                    break
            
    def read_shelf_label(self, position_index):
        """
        读取指定位置的标签槽中的物品名称
        Args:
            position_index: 标签槽位置索引 (0-3)
        Returns:
            str: 识别到的物品名称，如果无法识别则返回 None
        """
        if not 0 <= position_index < len(self.label_positions):
            raise ValueError("无效的标签槽位置索引")
            
        # 移动到标签槽位置
        x, y, z = self.label_positions[position_index]
        self.move_to_position(x, y)
        time.sleep(1)  # 等待相机稳定
        
        # 获取图像并识别文字
        image = self.vision_detector.get_camera_image()
        label_text = self.vision_detector.recognize_text(image)
        
        if label_text:
            # 将识别到的文字转换为物品代码
            for drink in self.shelf_drinks:
                if drink in label_text.lower():
                    return drink
                    
        return None

    def handle_first_level_drinks(self):
        """处理第一层饮料的上架任务"""
        placed_count = 0
        while placed_count < len(self.shelf_positions):
            # 读取当前标签槽的物品名称
            target_drink = self.read_shelf_label(placed_count)
            if not target_drink:
                print(f"无法识别第 {placed_count + 1} 个标签槽的物品名称")
                placed_count += 1
                continue
                
            # 获取图像
            image = self.vision_detector.get_camera_image()
            # 检测饮料
            detected_drinks = self.vision_detector.detect_drinks(image, [target_drink])
            
            for drink in detected_drinks:
                if drink['is_in_center']:
                    # 移动到饮料位置并抓取
                    self.move_and_grab(drink['position'])
                    # 移动到第三层指定位置并放置
                    self.move_and_place(self.shelf_positions[placed_count])
                    placed_count += 1
                    break
            
            time.sleep(0.5)  # 短暂延迟，避免过于频繁的检测

    def execute_task(self):
        """执行移动任务"""
        try:

            # 1. 识别初见物品
            # sample_item = self.observe_sample_item()
            # print(f"检测到的样品物品: {sample_item}")
            
            # 2. 收集第二层饮料
            self.collect_second_level_drinks()
            print(f"已收集的饮料: {self.collected_items}")
            
            # # 3. 处理第一层饮料
            # self.handle_first_level_drinks()
            
            # 4. 收集样品物品
            # self.collect_sample_items()
            # print(f"最终收集的物品: {self.collected_items}")
            
            # print("所有移动任务执行完成")
            
        except Exception as e:
            print(f"任务执行出错: {str(e)}")
        finally:
            # 关闭串口连接
            self.serial_comm.close()
