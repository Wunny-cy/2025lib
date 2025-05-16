from serial_communication import SerialCommunication
import time

floor = 1

class Robot:
    def __init__(self,detector):
        print("初始化中。。。")
        self.target_drinks = ["yykx", "wz", "bs", "yld"]#["营养快线", "旺仔", "百事", "养乐多"]
        self.shelf_drinks = ["riosmt", "dpty", "cp", "jdb"]#["锐澳水蜜桃", "东鹏", "茶π柠檬红茶", "加多宝"]
        self.label = ["锐澳", "东鹏", "茶Ⅱ", "加多宝"]#["锐澳水蜜桃", "东鹏", "茶π柠檬红茶", "加多宝"]
        self.collected_items = []
        self.grab_items = []
        self.placed_items = []
        self.vision_detector =detector  #VisionDetector()
        self.serial_comm = SerialCommunication()
        # self.DTG(1,1)
        print("DTG")
        # time.sleep(3)
        self.SVO(1, 50, 1000) 
        self.SVO(2, 15, 1000) 
        self.SVO(3, 85, 1000)
        # time.sleep(1.5)
        # self.SVO(1, 80, 1000) 
        # time.sleep(1.5)
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
        print(f"DTG{id}指令已发送")
        self.serial_comm.check_ok()
        print(f"DTG{id}指令已执行")
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
        print(f"SVO {id} {angle}指令已发送")
        self.serial_comm.check_ok()
        print(f"SVO {id} {angle}指令已执行")

    def arm_grab_item(self):
        """
        执行抓取动作
        """
        self.serial_comm.send_command("SVO 3 20 1000")  # 关闭夹爪
        print("关闭夹爪")
        self.serial_comm.check_ok()
        print(f"关闭夹爪指令已执行")

    def arm_release_item(self):
        """
        执行释放动作
        """
        self.serial_comm.send_command("SVO 3 50 1000")  # 打开夹爪
        print("打开夹爪")
        self.serial_comm.check_ok()
        print(f"打开夹爪指令已执行")

    def arm_grab(self):
        """用机械臂抓取物品"""
        # 打开机械爪
        self.arm_release_item()
        # 伸出电推杆
        self.DTG(3, 1)
        time.sleep(5)#延时等待电推杆到位        
        # 抓取物品
        self.arm_grab_item()
        # 略微抬起机械爪
        self.slide_move(1, 3)
        # 收回电推杆
        self.DTG(3, 0)
        time.sleep(5)#延时等待电推杆到位

    def arm_place(self):
        """用机械臂放置物品"""
        # 伸出电推杆
        self.DTG(3, 1)
        time.sleep(5)#延时等待电推杆到位  
        # 放置物品
        self.arm_release_item()
        # 收回电推杆
        self.DTG(3, 0)
        
    def arm_collect(self):
        """ 用机械臂收集物品 """
        self.DTG(3, 1)
        time.sleep(5)#延时等待电推杆到位  
        self.arm_grab_item()
        self.DTG(3, 0)
        time.sleep(5)#延时等待电推杆到位  
        self.arm_release_item()

    def hook_item1(self):
        """
        执行勾取动作
        """
        self.SVO(1, 125, 1000)
        time.sleep(1.5)
        self.SVO(1, 50, 1000)

    def hook_item2(self):
        """
        执行勾取动作
        """
        self.SVO(2, 80, 1000)
        self.serial_comm.check_ok()
        self.SVO(2, 15, 1000)
        self.serial_comm.check_ok()

    def slide_move(self , dir , turns):
        """
        滑轨定位控制
        Args:
            dir: 移动方向 (0:向下, 1:向上)
            turns: 移动圈数
        """
        self.serial_comm.send_command(f"MOVE 1 {dir} {turns}")  
        print(f"滑轨{dir} {turns}指令已发送")
        self.serial_comm.check_ok()
        print(f"滑轨{dir} {turns}指令已执行")
        time.sleep(1)

    def slide_floor_set1(self):
        """
        设置滑轨位置
        """
        self.serial_comm.send_command(f"SET1")
        print(f"回到零点并校正指令已发送")
        self.serial_comm.check_ok()
        print(f"回到零点并校正指令已执行")

    def slide_floor_set(self, position):
        """
        设置滑轨位置
        Args:
            floor: 楼层 (1,2,3)
        """
        if floor not in [1 ,2, 3] :
            raise ValueError("楼层为1，2，3")
        elif floor == position:
            raise ValueError("目标楼层和当前位置不能相同")
        self.serial_comm.send_command(f"F{floor}T{position}")
        print(f"滑轨从{floor}层到{position}层指令已发送")
        self.serial_comm.check_ok()
        print(f"滑轨从{floor}层到{position}层指令已执行")

    # def move_to_position(self, x, y):
    #     """
    #     控制机器人移动到指定位置
    #     Args:
    #         x: X坐标
    #         y: Y坐标
    #     """
    #     # 先移动X坐标
    #     self.move(f"X{x}")
    #     # 再移动Y坐标
    #     self.move(f"Y{y}")
    
    # def arm_up(self):
    #     print(1)#待修改

    def go_ahead(self):
        """
        控制机器人直行前进
        """
        # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
        command = f"FWD 500 150"
        self.serial_comm.send_command(command)
        print("前进指令已发送")
        self.serial_comm.check_ok()
        print("前进指令已执行")

    def travel(self):
        """
        控制机器人绕货架前进
        """
        # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
        command = f"ZX 500"
        self.serial_comm.send_command(command)
        print("前进检测指令已发送")
        self.serial_comm.check_ok()
        print("前进检测指令已执行")


    def slide_forward(self):
        """
        控制机器人前进
        """
        # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
        command = f"ZX 400"
        self.serial_comm.send_command(command)
        print("前进指令已发送")
        self.serial_comm.check_ok()
        print("前进指令已执行")

    def slide_backward(self):
        """
        控制机器人后退
        """
        # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
        command = f"HT 400"
        self.serial_comm.send_command(command)
        print("后退指令已发送")
        self.serial_comm.check_ok()
        print("后退指令已执行")
        

    def stop(self):
        """
        控制机器人停止运动
        """
        # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
        command = f"STOP"#待修改
        self.serial_comm.send_command(command)
        print("停止已发送")
        self.serial_comm.check_ok()
        print("已停止")

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
        # self.move_to_position(100, 100)
        # # 获取图像
        # image = self.vision_detector.get_camera_image()
        # 检测样品
        sample_item = self.vision_detector.detect_sample()
        if sample_item is None:
            raise Exception("未能检测到样品物品")
        return sample_item
        
    def collect_sample_items(self):
        """收集与模板匹配的物品"""
        collected_count = 0
        while collected_count < 2:
            # 检测与模板匹配的物品
            the_item = self.vision_detector.detect_sample_item()
            
            if the_item and the_item['confidence'] > 0.5:  # 设置置信度阈值
                x1, y1, x2, y2 = the_item['position']
                # 计算物品中心位置
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                if the_item['is_in_1']:
                    # 停止机器人
                    self.stop()
                    # 勾取2层饮料
                    self.hook_item1()
                    collected_count += 1 
                    print(f"已收集第 {collected_count} 个物品")
                elif the_item['is_in_2']:
                    # 停止机器人
                    self.stop()
                    # 勾取饮料
                    self.hook_item2()
                    collected_count += 1 
                    print(f"已收集第 {collected_count} 个物品")
                elif the_item['is_in_3']:
                    # 停止机器人
                    self.stop()
                    # 移动到3层并抓取
                    self.arm_collect()
                    collected_count += 1 
                    print(f"已收集第 {collected_count} 个物品")
                elif the_item['is_in_4']:
                    # 停止机器人
                    self.stop()
                    # 移动到1层并抓取
                    self.arm_up()
                    self.arm_collect()
                    collected_count += 1 
                    print(f"已收集第 {collected_count} 个物品")


            
            # 短暂延迟，避免过于频繁的检测
            time.sleep(0.1)
        
    def collect_second_level_drinks(self):
        """从第二层收集指定饮料"""
        # 控制机器人绕货架前进
        self.travel()
        print("前进")
        while len(self.collected_items) < len(self.target_drinks):
            image = self.vision_detector.get_camera_image()# 获取图像
            # 检测饮料
            detected_drinks = self.vision_detector.detect_drinks(image, self.target_drinks)
            # print(1)
            print(detected_drinks)
            for drink in detected_drinks:
                # print(detected_drinks)
                if drink['is_in_1'] and drink['name'] not in self.collected_items :
                    # 停止机器人
                    self.stop()
                    # 勾取饮料
                    self.hook_item1()
                    self.collected_items.append(drink['name'])
                    print(f"已收集饮料: {self.collected_items}")
                    self.collect_second_level_drinks()
                    break
                

    def handle_first_level_drinks(self):
        """将第一层待上架的饮料上架到第三层"""
        self.slide_move(1,1)
        self.slide_floor_set1()
        self.travel()
        print("前进")
        while len(self.placed_items) < len(self.shelf_drinks):
            image = self.vision_detector.get_camera_image()# 获取图像
            # 检测饮料
            detected_drinks = self.vision_detector.detect_drinks(image, self.shelf_drinks)
            print(f"检测到的饮料: {detected_drinks}")
            for drink in detected_drinks:
                # print(detected_drinks)
                if drink['is_in_4'] and drink['name'] not in self.placed_items :
                    self.stop()
                    self.arm_grab()
                    self.slide_floor_set(3)
                    time.sleep(5)
                    #放置
                    self.grabed_items_place(drink['name'])
                    self.placed_items.append(drink['name'])
                    print(f"已上架饮料: {self.placed_items}")
                    self.handle_first_level_drinks()

    def label_change(self, label):
        """
        设置需要更改的标签
        Args:
            label: 需要更改的标签
        """
        if label == "锐澳":
            return "riosmt"
        elif label == "东鹏":
            return "dpty"
        elif label.startswith("茶"):
            return "cp"
        elif label == "加多宝":
            return "jdb"
        else:
            return label  # 如果没有匹配的标签，返回原始标签
    
    # def name_to_label(self, name):
    #     """
    #     设置需要更改的标签
    #     Args:
    #         name: 需要更改的标签
    #     """
    #     if name == "riosmt":
    #         name = "锐澳"
    #     elif name == "dpty":
    #         name = "东鹏"
    #     elif name == "cp":
    #         name = "茶Ⅱ"
    #     elif name == "jdb":
    #         name = "加多宝"

    #     return name
    
    def grabed_items_place(self , label_name):
        """检测标签位置，将抓取的饮料放置到指定位置"""
        self.travel()
        print("前进")
        t = 0
        while t == 0:
            image = self.vision_detector.get_camera_image()# 获取图像
            # 检测标签
            detected_labels = self.vision_detector.detect_labels(image, self.label)
            for label in detected_labels:
                print(f"检测到的标签: {label['name']}")
                if label['ty'] and self.label_change(label['name']) == label_name and label['name'] not in self.placed_items :
                    if label['x_offset'] < 0:
                        self.slide_backward()
                    elif label['x_offset'] > 0:
                        self.slide_forward()
                    elif label['tx']:
                        #放置
                        self.stop()
                        self.arm_place()
                        t = 1
                        break
                break

    # def execute_task(self):
    #     """执行移动任务"""
    #     print("开始执行移动任务")
    #     try:
    #         # 1. 识别初见物品
    #         sample_item = self.observe_sample_item()
    #         print(f"检测到的样品物品: {sample_item}")
            
    #         # 2. 收集第二层饮料
    #         self.collect_second_level_drinks()
    #         print(f"已收集的饮料: {self.collected_items}")
            
    #         # 3. 处理第一层饮料
    #         self.handle_first_level_drinks()
            
    #         # 4. 收集样品物品
    #         self.collect_sample_items()
    #         print(f"最终收集的物品: {self.collected_items}")
            
    #         print("所有移动任务执行完成")
            
    #     except Exception as e:
    #         print(f"任务执行出错: {str(e)}")
    #     finally:
    #         # 关闭串口连接
    #         self.serial_comm.close()

    def execute_task1(self):
        """收集第二层饮料"""
        print("开始执行移动任务")
        try:
            # 2. 收集第二层饮料
            self.collect_second_level_drinks()
            print(f"已收集的饮料: {self.collected_items}")
            
            print("所有移动任务执行完成")
            
        except Exception as e:
            print(f"任务执行出错: {str(e)}")
        finally:
            # 关闭串口连接
            self.serial_comm.close()

    def execute_task2(self):
        """上架"""
        print("开始执行移动任务")
        try:
            self.handle_first_level_drinks()

            print("所有移动任务执行完成")
            
        except Exception as e:
            print(f"任务执行出错: {str(e)}")
        finally:
            # 关闭串口连接
            self.serial_comm.close()

    def execute_task3(self):
        """初见物品"""
        print("开始执行移动任务")
        try:
            self.go_ahead()
            # 识别初见物品
            sample_item = self.observe_sample_item()
            print(f"检测到的样品物品: {sample_item}")
            
            # 收集样品物品
            self.collect_sample_items()
            print(f"最终收集的物品: {self.collected_items}")
            
            print("所有移动任务执行完成")
            
        except Exception as e:
            print(f"任务执行出错: {str(e)}")
        finally:
            # 关闭串口连接
            self.serial_comm.close()

# if __name__ == "__main__":
#     robot = Robot()
#     robot.hook_item()
