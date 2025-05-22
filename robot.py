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
        self.FRONT_TOF_THRESHOLD = 300  # 前侧TOF安全距离阈值
        self.BACK_TOF_THRESHOLD = 300   # 后方TOF安全距离阈值
        self.slide_move(1,1)
        self.slide_floor_set1()
        self.SVO(2, 50, 1000) 
        # self.SVO(3, 15, 1000) 
        # self.SVO(4, 85, 1000)
        print("初始化完成")
        
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
        time.sleep(4)
        self.serial_comm.send_command(f"SAFE {id}")  # 停止电推杆
        print(f"SAFE {id}指令已发送")
        self.serial_comm.check_ok()
        print(f"SAFE {id}指令已执行")

    # def SAFE(self, id):
    #     """
    #     控制电推杆停止
    #     Args:
    #         id: 电推杆编号 (1-3)
    #     """
    #     self.serial_comm.send_command(f"SAFE {id}")  # 停止电推杆
    #     print(f"SAFE {id}指令已发送")
    #     self.serial_comm.check_ok()
    #     print(f"SAFE {id}指令已执行")

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
        self.serial_comm.send_command("SVO 4 25 1000")  # 关闭夹爪
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
        # time.sleep(4)#延时等待电推杆到位        
        # 抓取物品
        self.arm_grab_item()
        # 略微抬起机械爪
        self.slide_move(1, 3)
        # 收回电推杆
        self.DTG(3, 0)
        # time.sleep(4)#延时等待电推杆到位

    def arm_place(self):
        """用机械臂放置物品"""
        # 伸出电推杆
        self.DTG(3, 1)
        # time.sleep(4)#延时等待电推杆到位  
        # 放置物品
        self.arm_release_item()
        # 收回电推杆
        self.DTG(3, 0)
        
    def arm_collect(self):
        """ 用机械臂收集物品 """
        self.DTG(3, 1)
        # time.sleep(4)#延时等待电推杆到位  
        self.arm_grab_item()
        self.DTG(3, 0)
        # time.sleep(4)#延时等待电推杆到位  
        self.arm_release_item()

    def hook_item1(self):
        """
        执行勾取动作
        """
        self.SVO(2, 125, 1000)
        self.SVO(2, 50, 1000)

    def hook_item2(self):
        """
        执行勾取动作
        """
        self.SVO(3, 80, 1000)
        self.SVO(3, 25, 1000)

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
        
        while True:
            response = self.serial_comm.read_response()
            print(response)
            if response != None and "ok" in response.lower():
                break
            time.sleep(0.1)
        
        # 检查是否是有效的TOF数据响应
        if response.startswith("OK:F:"):
            # 解析格式为 "OK:F:1112,B:247,L:2531,R:195" 的响应
            # 提取每个方向的值
            tof_values = [1000, 1000, 1000, 1000]  # 默认值
            
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
            
        
        print("未能获取有效的TOF数据")
        return [1000, 1000, 1000, 1000]
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

    def travel(self, dir):
        """
        控制机器人绕货架前进
        """
        # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
        if dir == 1:
            command = f"ZX 500"
            self.serial_comm.send_command(command)
            print("前进运动指令已发送")
            self.serial_comm.check_ok()
            print("前进方向运动指令已执行")
        elif dir == 0:
            command = f"HT 500"
            self.serial_comm.send_command(command)
            print("后退运动指令已发送")
            self.serial_comm.check_ok()
            print("后退运动指令已执行")

    # def travel_slow(self, dir):
    #     """
    #     控制机器人绕货架前进
    #     """
    #     # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
    #     if dir == 1:
    #         command = f"ZX 500"
    #         self.serial_comm.send_command(command)
    #         print("前进运动指令已发送")
    #         self.serial_comm.check_ok()
    #         print("前进方向运动指令已执行")
    #     elif dir == 0:
    #         command = f"HT 500"
    #         self.serial_comm.send_command(command)
    #         print("后退运动指令已发送")
    #         self.serial_comm.check_ok()
    #         print("后退运动指令已执行")


    # def go_forward(self):
    #     """
    #     控制机器人前进
    #     """
    #     # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
    #     command = f"ZX 400"
    #     self.serial_comm.send_command(command)
    #     print("前进指令已发送")
    #     self.serial_comm.check_ok()
    #     print("前进指令已执行")

    # def go_backward(self):
    #     """
    #     控制机器人后退
    #     """
    #     # 定义一个字符串变量command，值为"SQUARE"，表示机器人前进的指令
    #     command = f"HT 400"
    #     self.serial_comm.send_command(command)
    #     print("后退指令已发送")
    #     self.serial_comm.check_ok()
    #     print("后退指令已执行")
        

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
        print("前进检测指令已发送")
        self.serial_comm.check_ok()
        print("前进检测指令已执行")
            
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
        if template_result and template_result['confidence'] > 0.2:  # 设置置信度阈值
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
        
    def collect_second_level_drinks(self,dir,t):
        """从第二层收集指定饮料"""
        # 控制机器人绕货架前进
        self.travel(dir)
        print(f"绕货架{dir}前进")
        while len(self.collected_items) < len(self.target_drinks) and t<3:
            # 读取TOF传感器数据
            tof_values = self.read_tof_list()
            front_tof = tof_values[0]
            back_tof = tof_values[1]
            # 获取图像
            image = self.vision_detector.get_camera_image()
            # 检测饮料
            detected_drinks = self.vision_detector.detect_drinks(image, self.target_drinks)
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
                
                # 递归调用并处理返回值
                result_dir = self.collect_second_level_drinks(dir,t)
                return result_dir, t  # 将递归调用的结果向上传递

            if dir == 1 and front_tof <= self.FRONT_TOF_THRESHOLD:
                    print(f"前方TOF值 {front_tof} <= {self.FRONT_TOF_THRESHOLD}，停止前进")
                    self.stop()
                    dir = 1-dir
                    self.travel(dir)
                    t = t + 1
                    print(f'1t: {t}')
            elif dir == 0 and back_tof <= self.BACK_TOF_THRESHOLD:
                    print(f"后方TOF值 {back_tof} <= {self.BACK_TOF_THRESHOLD}，停止后退")
                    self.stop()
                    dir = 1-dir
                    self.travel(dir)
                    t = t + 1
                    print(f'1t: {t}')

            if t >= 2:
                print(f'2t: {t}')
                t = 0
                return dir,t
            
        return dir,t
            
                
    def handle_first_level_drinks(self, dir, t):
        """将第一层待上架的饮料上架到第三层"""
        self.travel(dir)
        print(111111111)
        while len(self.placed_items) < len(self.shelf_drinks) and t<3:
            # 读取TOF传感器数据
            tof_values = self.read_tof_list()
            front_tof = tof_values[0]
            back_tof = tof_values[1]
            # 获取图像
            image = self.vision_detector.get_camera_image()
            # 检测饮料
            detected_drinks = self.vision_detector.detect_drinks(image, self.shelf_drinks)
            print(f"检测到的饮料: {detected_drinks}")
            if dir == 1 and front_tof <= self.FRONT_TOF_THRESHOLD:
                    print(f"前方TOF值 {front_tof} <= {self.FRONT_TOF_THRESHOLD}，停止前进")
                    self.stop()
                    dir = 1-dir
                    self.travel(dir)
                    t += 1
                    print(f'1t: {t}')
            elif dir == 0 and back_tof <= self.BACK_TOF_THRESHOLD:
                    print(f"后方TOF值 {back_tof} <= {self.BACK_TOF_THRESHOLD}，停止后退")
                    self.stop()
                    dir = 1-dir
                    self.travel(dir)
                    t += 1
                    print(f'1t: {t}')
            else:
                for drink in detected_drinks:
                    # print(detected_drinks)
                    if drink['is_in_4'] and drink['name'] not in self.placed_items:
                        self.stop()
                        self.arm_grab()
                        # self.MOVE(2, 1000, 20)
                        self.slide_floor_set(3)
                        time.sleep(5)
                        dir, t=self.grabed_items_place(drink['name'], dir, t) #放置
                        print(f"已上架饮料: {self.placed_items}")
                        dir, t=self.handle_first_level_drinks(dir, t)
                        
            if t >= 2:
                print(f'2t: {t}')
                return dir, t
                
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
        
    
    def grabed_items_place(self , label_name_EN, dir, t):
        """检测标签位置，将抓取的饮料放置到指定位置"""
        # self.travel(dir)
        p = 0#条件
        image = self.vision_detector.get_camera_image()# 获取图像
        # 检测标签
        detected_labels = self.vision_detector.detect_labels(image, self.label, label_name_EN)
        for label in detected_labels:
            print(f"检测到的标签: {label['name']}")
            if label['ty'] and self.label_change_CHtoEN(label['name']) == label_name_EN and label['name'] not in self.placed_items and 10 < label['x_offset'] < 10:
                #放置
                print(33333)
                cv2.imwrite('label.jpg', image)
                self.stop()
                print("开始矫正")
                cv2.imwrite('label1.jpg', image)
                self.grabed_items_place_correct(label['name'])#输入中文标签
                self.arm_place()
                self.placed_items.append(label['name'])
                self.slide_floor_set1()
                p = 1
                break
        if p == 0:
            self.travel(dir)
        while p == 0:
            print(222222)
            # 读取TOF传感器数据
            tof_values = self.read_tof_list()
            front_tof = tof_values[0]
            back_tof = tof_values[1]
            if dir == 1 and front_tof <= self.FRONT_TOF_THRESHOLD:
                print(f"前方TOF值 {front_tof} <= {self.FRONT_TOF_THRESHOLD}，停止前进")
                self.stop()
                dir = 1-dir
                self.travel(dir)
                t = t + 1
                print(f'1t: {t}')
            elif dir == 0 and back_tof <= self.BACK_TOF_THRESHOLD:
                print(f"后方TOF值 {back_tof} <= {self.BACK_TOF_THRESHOLD}，停止后退")
                self.stop()
                dir = 1-dir
                self.travel(dir)
                t += t + 1
                print(f'1t: {t}')
            else:
                image = self.vision_detector.get_camera_image()# 获取图像
                # 检测标签
                detected_labels = self.vision_detector.detect_labels(image, self.label, label_name_EN)
                for label in detected_labels:
                    print(f"检测到的标签: {label['name']}")
                    if label['ty'] and self.label_change_CHtoEN(label['name']) == label_name_EN and label['name'] not in self.placed_items :
                        if label['x_offset'] < -10:
                            self.travel(0)
                            # x_last=label['x_offset']
                        elif label['x_offset'] > 10:
                            self.travel(1)
                            # x_last=label['x_offset']
                        else :
                            #放置
                            print(33333)
                            cv2.imwrite('label.jpg', image)
                            self.stop()
                            print("开始矫正")
                            cv2.imwrite('label1.jpg', image)
                            # x_last -= label['x_offset']
                            if t != 0:
                                self.grabed_items_place_correct(label['name'])#输入中文标签
                            else:
                                self.MOVE(1,500,120)
                            self.arm_place()
                            self.placed_items.append(label['name'])
                            self.slide_floor_set1()
                            p = 1
            
        return dir,t

    def grabed_items_place_correct(self , label_name):
        image = self.vision_detector.get_camera_image()# 获取图像
        print(11111)
        # 检测标签
        detected_labels = self.vision_detector.detect_labels(image, self.label, label_name)
        cv2.imwrite('label2.jpg', image)
        print(22222222222)
        f = 0.4
        for label in detected_labels:
            print(f"二次检测到的标签: {label['name']}")
            if label['x_offset'] < -10:
                self.MOVE(1,500,label['x_offset']*(-f)+50)
                print(555555555555)
                print(label['x_offset']*(-f)+50)
                self.MOVE(1,500,120)
                print('前移')
            elif label['x_offset'] > 10:
                self.MOVE(0,500,label['x_offset']*f+50)
                print(label['x_offset']*(-f)+50)
                self.MOVE(1,500,120)
                print('后移')
            else:
                self.MOVE(1,500,120)
                #放置
        self.stop()

    def execute_task(self):
        """执行移动任务"""
        print("开始执行移动任务")
        try:
            self.MOVE(2, 1000, 210)
            dir=0
            t=0
            p=0
            # dir, t = self.collect_second_level_drinks(dir, t)
            print(555555555)
            p=p+t
            t=0
            dir, t = self.handle_first_level_drinks(dir, t)
            print(666666666)
            p=p+t
            t=0
            if p ==2 or p==4 or p==0:
                self.DTG(1,0)
                self.RT(90)
                self.MOVE(0,1000,2000)
                self.RT(90)
                dir=1
            elif p == 1 or p == 3:
                self.DTG(1,0)
                self.RT(-90)
                self.MOVE(1,1000,2000)
                self.RT(-90)
                dir=0
            t=0
            dir, t = self.collect_second_level_drinks(dir, t)
            print(555555555)
            p=p+t
            t=0
            dir, t = self.handle_first_level_drinks(dir, t)
            print(666666666)
            p=p+t
            t=0

            self.stop()
            
        except Exception as e:
            print(f"任务执行出错: {str(e)}")
        finally:
            print("移动任务执行完成，准备继续移动至指定位置")
            self.stop()

    def execute_task1(self, dir):
        """收集第二层饮料
        Args:
            dir: 移动方向 (1-前进收集, 0-后退收集)
        """
        print("开始执行移动任务")
        try:
            # while len(self.collected_items) < len(self.target_drinks):
                # 收集第二层饮料，传递方向参数
            t=0
            self.collect_second_level_drinks(dir, t)
                # print(f"已收集的饮料: {self.collected_items}")
            self.stop()
                # print("移动任务执行完成，准备继续移动至指定位置")
                
                # # 定义TOF索引和阈值
                # FRONT_TOF_INDEX = 0  # 前侧TOF在列表中的索引，请根据实际情况调整
                # BACK_TOF_INDEX = 1   # 后方TOF在列表中的索引，请根据实际情况调整
                
                # self.FRONT_TOF_THRESHOLD = 250  # 前侧TOF安全距离阈值
                # self.BACK_TOF_THRESHOLD = 550   # 后方TOF安全距离阈值
                
                # # 根据方向继续移动直至达到指定TOF值
                # if dir == 1:
                #     # 前进模式
                #     print("继续前进直至前方TOF值小于等于300")
                #     while True:
                #         # 获取当前前方TOF值
                #         tof_values = self.read_tof_list()
                #         front_tof = tof_values[FRONT_TOF_INDEX]
                        
                #         if front_tof <= self.FRONT_TOF_THRESHOLD:
                #             print(f"前方TOF值 {front_tof} <= {self.FRONT_TOF_THRESHOLD}，停止前进")
                #             self.stop()
                #             break
                        
                #         # 继续前进
                #         self.go_forward()
                #         time.sleep(0.1)  # 短暂延时，避免频繁读取TOF值
                # else:
                #     # 后退模式
                #     print("继续后退直至后方TOF值小于等于600")
                #     while True:
                #         # 获取当前后方TOF值
                #         tof_values = self.read_tof_list()
                #         back_tof = tof_values[BACK_TOF_INDEX]
                        
                #         if back_tof <= self.BACK_TOF_THRESHOLD:
                #             print(f"后方TOF值 {back_tof} <= {self.BACK_TOF_THRESHOLD}，停止后退")
                #             self.stop()
                #             break
                        
                #         # 继续后退
                #         self.go_backward()
                #         time.sleep(0.1)  # 短暂延时，避免频繁读取TOF值
                # dir = 1-dir
                # self.execute_task1(dir)
            print("所有任务执行完成")
            self.stop()
            
        except Exception as e:
            print(f"任务执行出错: {str(e)}")
            self.stop()  # 确保出错时停车


    def execute_task2(self, dir):
        """上架饮料
        Args:
            dir: 移动方向 (1-前进, 0-后退)
        """
        print("开始执行移动任务")
        try:
            # self.slide_move(1,1)
            t=0
            self.handle_first_level_drinks(dir, t)

            print("所有移动任务执行完成")
            self.stop()
            
        except Exception as e:
            print(f"任务执行出错: {str(e)}")

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
