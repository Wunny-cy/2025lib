from serial_communication import SerialCommunication
import time
import cv2

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
        # self.last = 1
        # self.DTG(1,1)
        # print("DTG")
        # time.sleep(3)
        # self.SVO(2, 50, 1000) 
        # self.SVO(3, 15, 1000) 
        self.SVO(4, 85, 1000)
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
        print(f"DTG{id} {dir}指令已发送")
        self.serial_comm.check_ok()
        print(f"DTG{id} {dir}指令已执行")
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
        self.serial_comm.send_command("SVO 4 20 1000")  # 关闭夹爪
        print("关闭夹爪")
        self.serial_comm.check_ok()
        print(f"关闭夹爪指令已执行")

    def arm_release_item(self):
        """
        执行释放动作
        """
        self.serial_comm.send_command("SVO 4 50 1000")  # 打开夹爪
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
        self.SVO(2, 125, 1000)
        time.sleep(1.5)
        self.SVO(2, 50, 1000)

    def hook_item2(self):
        """
        执行勾取动作
        """
        self.SVO(3, 80, 1000)
        self.serial_comm.check_ok()
        self.SVO(3, 15, 1000)
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
    def read_tof_list(self):
        """
        读取TOF传感器值，返回值为列表形式
        
        Returns:
            list: 包含四个方向TOF值的列表 [前, 后, 左, 右]，读取失败则返回[-1, -1, -1, -1]
        """
        # 发送GD指令读取TOF传感器值
        self.serial_comm.send_command("GD")
        
        # 等待有效响应
        max_attempts = 3
        attempts = 0
        
        while attempts < max_attempts:
            response = self.serial_comm.read_response()
            
            # 检查是否是有效的TOF数据响应
            if response.startswith("OK:F:"):
                try:
                    # 解析格式为 "OK:F:1112,B:247,L:2531,R:195" 的响应
                    # 提取每个方向的值
                    tof_values = [-1, -1, -1, -1]  # 默认值
                    
                    # 分割响应字符串
                    parts = response.strip().split(',')
                    
                    for part in parts:
                        if part.startswith("OK:F:"):
                            tof_values[0] = int(part.split(':')[2])
                        elif part.startswith("B:"):
                            tof_values[1] = int(part.split(':')[1])
                        elif part.startswith("L:"):
                            tof_values[2] = int(part.split(':')[1])
                        elif part.startswith("R:"):
                            tof_values[3] = int(part.split(':')[1])
                    
                    print(f"TOF传感器值: 前={tof_values[0]}mm, 后={tof_values[1]}mm, 左={tof_values[2]}mm, 右={tof_values[3]}mm")
                    return tof_values
                    
                except (ValueError, IndexError) as e:
                    print(f"解析TOF值出错: {e}")
                    attempts += 1
            else:
                # 不是有效的TOF数据响应，继续等待
                attempts += 1
                time.sleep(0.1)
        
        print("未能获取有效的TOF数据")
        return [-1, -1, -1, -1]
    def MOVE(self, direction, speed, distance):
        """
        控制机器人移动
        Args:
            direction: 移动方向 (0-前进, 1-后退, 2-左移, 3-右移)
            speed: 移动速度 (1-1000，表示速度)
            distance: 移动距离 (单位：毫米)
        """
        # 方向映射
        direction_map = {
            0: "FWD",  # 前进
            1: "BWD",  # 后退
            2: "LFT",  # 左移
            3: "RGT"   # 右移
        }
        
        # 验证方向参数
        if direction not in direction_map:
            raise ValueError("无效的方向，必须是0(前进)、1(后退)、2(左移)或3(右移)")
        
        # 获取方向指令
        direction_cmd = direction_map[direction]
        
        # 验证速度参数
        if not 1 <= speed <= 1000:
            raise ValueError("无效的速度值，必须在1-1000之间")
        
        # 验证距离参数
        if distance <= 0:
            raise ValueError("无效的距离值，必须大于0")
        
        # 发送移动指令
        self.serial_comm.send_command(f"{direction_cmd} {speed} {distance}")
        print(f"{direction_cmd} {speed} {distance}指令已发送")
        self.serial_comm.check_ok()
        print(f"{direction_cmd} {speed} {distance}指令已执行")

    # def go_ahead(self):
    #     """
    #     控制机器人直行前进
    #     """
    #     # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
    #     command = f"FWD 500 1100"
    #     self.serial_comm.send_command(command)
    #     print("前进指令已发送")
    #     self.serial_comm.check_ok()
    #     print("前进指令已执行")

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


    def go_forward(self):
        """
        控制机器人前进
        """
        # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
        command = f"ZX 400"
        self.serial_comm.send_command(command)
        print("前进指令已发送")
        self.serial_comm.check_ok()
        print("前进指令已执行")

    def go_backward(self):
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
        # 获取图像
        # image = self.vision_detector.get_camera_image()
        # 检测样品
        image = self.vision_detector.get_camera_image()# 获取图像
        sample_item = self.vision_detector.detect_sample(image)
        sample_item1 = self.vision_detector.detect_sample1(image)
        if sample_item != False:
            print("边缘检测方法检测到样品物品")
            return sample_item
        elif sample_item1 != False:
            print("YOLO方法检测到样品物品")
            return sample_item1
        else:
            print("寄了 这分给了")
            return False
        
        
    def collect_sample_items(self, template_result):
        """收集与模板匹配的物品"""
        if template_result and template_result['confidence'] > 0.5:  # 设置置信度阈值
            if template_result['is_in_1']:
                # 停止机器人
                self.stop()
                # 勾取2层饮料
                self.hook_item1()
                collected_count += 1 
                print(f"已收集第 {collected_count} 个物品")
            elif template_result['is_in_2']:
                # 停止机器人
                self.stop()
                # 勾取饮料
                self.hook_item2()
                collected_count += 1 
                print(f"已收集第 {collected_count} 个物品")
            elif template_result['is_in_3']:
                # 停止机器人
                self.stop()
                # 移动到3层并抓取
                self.arm_collect()
                collected_count += 1 
                print(f"已收集第 {collected_count} 个物品")
            elif template_result['is_in_4']:
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
                    print(f"已上架饮料: {self.placed_items}")
                    self.handle_first_level_drinks()

    def label_change_CHtoEN(self, label):
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
        
    
    def grabed_items_place(self , label_name_EN):
        """检测标签位置，将抓取的饮料放置到指定位置"""
        self.travel()
        print("前进")
        t = 0
        while t == 0:
            image = self.vision_detector.get_camera_image()# 获取图像
            # 检测标签
            detected_labels = self.vision_detector.detect_labels(image, self.label, label_name_EN)
            for label in detected_labels:
                print(f"检测到的标签: {label['name']}")
                if label['ty'] and self.label_change_CHtoEN(label['name']) == label_name_EN and label['name'] not in self.placed_items :
                    if label['x_offset'] < 0:
                        self.go_backward()
                        # self.last=-1
                    elif label['x_offset'] > 0:
                        self.go_forward()
                        # self.last=1
                    elif label['tx']:
                        #放置
                        print(33333)
                        cv2.imwrite('label.jpg', image)
                        self.stop()
                        print("开始矫正")
                        self.grabed_items_place_correct(label['name'])#输入中文标签
                        self.arm_place()
                        self.placed_items.append(label['name'])
                        t = 1
                        break
                break

    def grabed_items_place_correct(self , label_name):
        image = self.vision_detector.get_camera_image()# 获取图像
        print(11111)
        # 检测标签
        detected_labels = self.vision_detector.detect_labels(image, self.label, label_name)
        print(22222)
        f = 0.4
        for label in detected_labels:
            print(f"二次检测到的标签: {label['name']}")
            if label['ty'] and label['name'] == label_name and label['name'] not in self.placed_items :
                if label['x_offset'] < 0:
                    self.move(label['x_offset']*f,0)
                    print('右移')
                elif label['x_offset'] > 0:
                    self.move(label['x_offset']*f,0)
                    print('左移')
                elif label['tx']:
                    print("grabed_items_place_correct 6666")
                    #放置
                    self.stop()


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
            self.stop()

    def execute_task2(self):
        """上架"""
        print("开始执行移动任务")
        try:
            self.slide_move(1,1)
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
            self.MOVE(0,1000,1100)
            # 识别初见物品1
            sample_item1 = self.observe_sample_item()
            if sample_item1 == False:
                raise Exception("未检测到初见物品")
            else :
                cv2.imwrite('template1.jpg', sample_item1)
                print(f"检测到的样品物品1: {sample_item1}")
            
            self.MOVE(0,1000,1100)
            self.RT(90)
            self.MOVE(0,1000,2000)
            self.RT(90)
            self.MOVE(0,1000,1100)
            # 识别初见物品2
            sample_item2 = self.observe_sample_item()
            if sample_item2 == False:
                raise Exception("未检测到初见物品")
            else :
                cv2.imwrite('template2.jpg', sample_item2)
                print(f"检测到的样品物品2: {sample_item2}")
            
            collected_count = 0
            # 读取模板图像
            template1 = cv2.imread('template1.jpg', cv2.IMREAD_GRAYSCALE)
            template2 = cv2.imread('template2.jpg', cv2.IMREAD_GRAYSCALE)
            if template1 is None and template2 is None:
                raise Exception("无法读取模板图像")
            while collected_count < 2:
                # 检测与模板匹配的物品
                the_item = self.vision_detector.detect_sample_item(template1, template2)
                template1_result = the_item['template1']
                template2_result = the_item['template2']
                # 收集样品物品
                if template1_result== True:
                    self.collect_sample_items(template1_result)
                    break
                elif template2_result == True:
                    self.collect_sample_items(template2_result)
                    break
            print(f"最终收集的物品: {self.collected_items}")
            
        except Exception as e:
            print(f"任务执行出错: {str(e)}")

# if __name__ == "__main__":
#     robot = Robot()
#     robot.hook_item()
