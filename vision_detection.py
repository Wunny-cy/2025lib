import os
os.environ['YOLO_VERBOSE'] = str(False)
from ultralytics import YOLO
import cv2
import numpy as np
import torch
# from paddleocr import PaddleOCR

regions = []
class VisionDetector:
    def __init__(self):
        self.model = YOLO('best.pt')  
        self.image_center_threshold = 50  # 图像中心区域的阈值（像素）
        
        # 检查是否有可用的GPU
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        self.model.to(self.device)  # 将模型转移到指定的设备上
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5000)  # 设置一个很大的值
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5000)
        
        # 获取实际设置的分辨率
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头分辨率: {actual_width}x{actual_height}")
        
        # 创建可调整大小的窗口，并保持图像比例
        cv2.namedWindow('Model Test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow('Model Test', 750, 1440)  # 设置初始窗口大小
        
        # 后续的初始化操作可以在这里执行
        self.after_initialization()

    def start_detection(self):
        try:
            while True:
                ret, frame = self.cap.read()

                if not ret:
                    raise Exception("无法读取图像")
                
                # 逆时针旋转90度
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                # 获取图像中心点
                height, width = frame.shape[:2]
                center_x = width // 2
                center_y = height // 2
                
                # 绘制中心区域矩形框（蓝色）
                x1 = center_x - 120
                y1 = center_y + 180
                x2 = center_x + 120
                y2 = center_y - 200
                # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色矩形框
                
                # 绘制第二层拨杆区域矩形框（蓝色）
                x11 =30
                y11 = 1000
                x12 = 230
                y12 = 1500
                cv2.rectangle(frame, (x11, y11), (x12, y12), (255, 0, 0), 5)  # 蓝色矩形框

                # 绘制下层区域矩形框（蓝色）
                # x21 = 550
                # y21 = 1460
                # x22 = 880
                # y22 = 1080
                # cv2.rectangle(frame, (x21, y21), (x22, y22), (255, 0, 0), 2)  # 蓝色矩形框

                # 计算九宫格分割线位置
                # 水平分割线
                top_y = 0
                middle_y1 = y1
                middle_y2 = y2
                bottom_y = height
                
                # 垂直分割线
                left_x = 0
                middle_x1 = x1
                middle_x2 = x2
                right_x = width

                # 绘制水平分割线
                cv2.line(frame, (0, middle_y1), (width, middle_y1), (0, 255, 0), 3)
                cv2.line(frame, (0, middle_y2), (width, middle_y2), (0, 255, 0), 3)
                
                # 绘制垂直分割线
                cv2.line(frame, (middle_x1, 0), (middle_x1, height), (0, 255, 0), 3)
                cv2.line(frame, (middle_x2, 0), (middle_x2, height), (0, 255, 0), 3)

                # 生成九个区域的列表
                global  regions
                regions.clear()
                # 上排三个区域
                regions.append((left_x, top_y, middle_x1, middle_y1))
                regions.append((middle_x1, top_y, middle_x2, middle_y1))
                regions.append((middle_x2, top_y, right_x, middle_y1))
                
                # 中排三个区域
                regions.append((left_x, middle_y1, middle_x1, middle_y2))
                regions.append((middle_x1, middle_y1, middle_x2, middle_y2))  # 中间区域
                regions.append((middle_x2, middle_y1, right_x, middle_y2))
                
                # 下排三个区域
                regions.append((left_x, middle_y2, middle_x1, bottom_y))
                regions.append((middle_x1, middle_y2, middle_x2, bottom_y))
                regions.append((middle_x2, middle_y2, right_x, bottom_y))
                # 运行模型检测
                results = self.model.predict(frame)

                # 在图像上绘制检测结果
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        # 获取置信度和类别
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        
                        # 绘制边界框
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # 添加标签文本
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(frame, label, (int(x1), int(y1-10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # 显示图像
                cv2.imshow('Model Test', frame)
                
                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # 释放资源
            cv2.destroyAllWindows()
            self.cap.release()

    def after_initialization(self):
        # 这里可以添加后续的初始化操作
        print("执行后续初始化操作")
        # 示例：获取图像
        # image = self.get_camera_image()
        # 示例：检测样品
        # sample_item = self.detect_sample()
        # print(f"检测到的样品: {sample_item}")
        # 示例：检测饮料
        # target_drinks = ["yykx", "wz", "bs", "yld"]
        # detected_drinks = self.detect_drinks(image, target_drinks)
        # print(f"检测到的饮料: {detected_drinks}")

    def convert_np_floats(self,obj):
        """
        将numpy.float32类型转换为Python float类型
        Returns:
            Python float类型
        """
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, tuple):
            return tuple(self.convert_np_floats(item) for item in obj)
        elif isinstance(obj, list):
            return [self.convert_np_floats(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: self.convert_np_floats(v) for k, v in obj.items()}
        else:
            return obj

    def get_camera_image(self):
        """
        获取摄像头图像并实时显示（最高分辨率）
        Returns:
            numpy.ndarray: 图像数据
        """
        ret, frame = self.cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame
        # try:
            
        #     while True:
        #         ret, frame = self.cap.read()
        #         if not ret:
        #             raise Exception("无法读取图像")
                
            
        #         # cv2.imshow('Camera', frame)
                
        #         # 按'q'键退出
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break
                
        #     return frame
        
        # finally:
        #     cv2.destroyAllWindows()

    def save_camera_image(self):
        """
        保存摄像头图像
        Returns:
            图像路径
        """
        ret, frame = self.cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        image_path = 'captured_image.png'
        cv2.imwrite(image_path, frame)
        return image_path
    
    # def recognize_shopping_list(self,image_path):
    #     ocr = PaddleOCR(use_angle_cls=True)  # 启用角度分类
    #     result = ocr.ocr(image_path)
    #     if not result:  # 现在这个检查只确保result不是空列表
    #         print("没有识别到任何内容")
    #         return []

    #     recognized_items = []  # 存储识别结果的数组
    #     if result is not None:  # 添加检查确保result不是None
    #         for line in result:
    #             if line:  # 确保line本身不为空
    #                 for element in line:
    #                     if isinstance(element, list) and len(element) > 1:
    #                         box = element[0]  # 包围盒坐标
    #                         text = element[1][0]  # 识别的文本
    #                         # 计算中心坐标
    #                         center_x = sum([point[0] for point in box]) / 4
    #                         center_y = sum([point[1] for point in box]) / 4
    #                         # 将文本和中心坐标存储为元组，并添加到数组中
    #                         recognized_items.append([text, (center_x, center_y)])
    #     return recognized_items
    
    def replace_text_with_numbers(self,items):
        for i in range(len(items)):
            item = items[i]
            if item[0] == 'I':
                items[i][0] = 1
            elif item[0] == '1':
                items[i][0] = 1
            elif item[0] == '2':
                items[i][0] = 2
            elif item[0] == '7':
                items[i][0] = 2
            elif item[0] == '乙':
                items[i][0] = 2
            elif item[0] == '3':
                items[i][0] = 3
            elif item[0] == '8':
                items[i][0] = 3
        return items
    
    def final(self, items, items_keywords):
        """
        处理识别结果，将识别结果中的数字与物品名称对应
        Args:
            items: 识别结果，每个元素为 (物品名称, 中心坐标)
        Returns:
            list: 最终结果，每个元素为 (物品名称, 数量)
        """
        final_items = []  # 定义一个空列表，用于存储最终结果
        for i in range(len(items)):
            item_name = items[i][0]  # 访问第一个元素（物品名称）
            item_y = items[i][1][1]  # 访问第二个元素（坐标）
            if item_name in items_keywords :
                for j in range(len(items)):
                    num_name = items[j][0]  # 访问第一个元素（物品名称）
                    num_y = items[j][1][1]
                    if abs(num_y - item_y) < 15 and isinstance(num_name, int):
                        final_items.append((item_name, num_name))
                    else:
                        final_items.append((item_name, 1))
                        break
        return final_items

    
    def get_image_center(self, image):
        """
        获取图像中心位置
        Args:
            image: 输入图像
        Returns:
            tuple: (center_x, center_y)
        """
        height, width = image.shape[:2]
        return (width // 2, height // 2)
    
    def is_in_center(self, x_offset, y_offset):
        """
        判断目标是否在中心区域
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
            image: 输入图像
        Returns:
            tuple: (x_offset, y_offset) 偏移量
                  x_offset: 正值表示目标在中心区域右侧，负值表示目标在中心区域左侧
                  y_offset: 正值表示目标在中心区域下方，负值表示目标在中心区域上方
        """
        if -3 < x_offset < 3 and -3 < y_offset < 3 :
            return True
        else:
            return False
    
    
    
    def is_in_1(self, bbox):
        """
        判断目标是否在1区域
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
            image: 输入图像
        Returns:
            float: 偏移量，正值表示目标在中心区域右侧，负值表示目标在中心区域左侧
        """
        # 获取图像中心点
        # height, width = image.shape[:2]  # 获取图像的高度和宽度
        # center_x = width // 2
        x1, y1, x2, y2 = bbox
        # 定义中心区域
        x11 =30
        y11 = 1000
        x12 = 230
        y12 = 1500
        
        # 计算目标物体的中心点
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        
        # 计算相对于中心区域的偏移量
        x_offset = 0
        y_offset = 0
        
        if x11 < target_center_x < x12 and y11 < target_center_y < y12:
            return True
        else:
            return False

        # # 计算水平偏移
        # if target_center_x < x11:
        #     x_offset = target_center_x - x11  # 负值，表示在左侧
        #     # return True
        # elif target_center_x > x12:
        #     x_offset = target_center_x - x12  # 正值，表示在右侧
        #     # return False
        
        # if target_center_y < y11:
        #     y_offset = target_center_y - y11  # 负值，表示在上方
        #     # return True
        # elif target_center_y > y12:
        #     y_offset = target_center_y - y12

        # return (x_offset, y_offset)
        # print("x_offset:", x_offset)
        # return x_offset
    
    def is_in_2(self, bbox):
        """
        计算边界框相对于1区域的偏移量
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
            image: 输入图像
        Returns:
            float: 偏移量，正值表示目标在中心区域右侧，负值表示目标在中心区域左侧
        """
        x1, y1, x2, y2 = bbox
        # 定义中心区域(待定)
        x21 =30
        y21 = 1000
        x22 = 230
        y22 = 1500
        
        # 计算目标物体的中心点
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        
        # 计算相对于中心区域的偏移量
        if target_center_x < x21:
            x_offset = target_center_x - x21  # 负值，表示在左侧
        elif target_center_x > x22:
            x_offset = target_center_x - x22  # 正值，表示在右侧
        
        if target_center_y < y21:
            y_offset = target_center_y - y21  # 负值，表示在上方
        elif target_center_y > y22:
            y_offset = target_center_y - y22  # 正值，表示在下方

        return (x_offset, y_offset)
    
    def is_in_3(self, bbox):
        """
        计算边界框相对于1区域的偏移量
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
            image: 输入图像
        Returns:
            float: 偏移量，正值表示目标在中心区域右侧，负值表示目标在中心区域左侧
        """
        x1, y1, x2, y2 = bbox
        # 定义中心区域(待定)
        x21 =30
        y21 = 1000
        x22 = 230
        y22 = 1500
        
        # 计算目标物体的中心点
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        
        # 计算相对于中心区域的偏移量
        if target_center_x < x21:
            x_offset = target_center_x - x21  # 负值，表示在左侧
        elif target_center_x > x22:
            x_offset = target_center_x - x22  # 正值，表示在右侧
        
        if target_center_y < y21:
            y_offset = target_center_y - y21  # 负值，表示在上方
        elif target_center_y > y22:
            y_offset = target_center_y - y22  # 正值，表示在下方

        return (x_offset, y_offset)
    
    def is_in_4(self, bbox):
        """
        计算边界框相对于1区域的偏移量
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
            image: 输入图像
        Returns:
            float: 偏移量，正值表示目标在中心区域右侧，负值表示目标在中心区域左侧
        """
        x1, y1, x2, y2 = bbox
        # 定义中心区域(待定)
        x21 =30
        y21 = 1000
        x22 = 230
        y22 = 1500
        
        # 计算目标物体的中心点
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        
        # 计算相对于中心区域的偏移量
        if target_center_x < x21:
            x_offset = target_center_x - x21  # 负值，表示在左侧
        elif target_center_x > x22:
            x_offset = target_center_x - x22  # 正值，表示在右侧
        
        if target_center_y < y21:
            y_offset = target_center_y - y21  # 负值，表示在上方
        elif target_center_y > y22:
            y_offset = target_center_y - y22  # 正值，表示在下方

        return (x_offset, y_offset)

    

    def detect_sample(self):
        """
        提取图像中心的物品并保存为模板图片
        Returns:
            bool: 是否成功保存模板
        """
        try:
            # 获取摄像头图像
            ret, frame = self.cap.read()
            if not ret:
                print("错误：无法读取摄像头图像")
                return False

            # 获取图像中心区域
            height, width = frame.shape[:2]
            center_x = width // 2
            center_y = height // 2
            
            # 定义中心区域的大小（可以根据需要调整）
            region_size = 200  # 中心区域边长的一半
            
            # 计算中心区域的边界
            x1 = max(0, center_x - region_size)
            y1 = max(0, center_y - region_size)
            x2 = min(width, center_x + region_size)
            y2 = min(height, center_y + region_size)
            
            # 提取中心区域
            center_region = frame[y1:y2, x1:x2]
            
            # 转换为灰度图
            gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
            
            # 应用双边滤波
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # 应用自适应阈值处理
            thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # 查找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                print("警告：未在中心区域找到物品")
                return False
            
            # 找到最大的轮廓
            max_contour = max(contours, key=cv2.contourArea)
            
            # 获取轮廓的边界框
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # 在原图上绘制边界框
            cv2.rectangle(center_region, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 提取物品区域
            item_region = center_region[y:y+h, x:x+w]
            
            # 保存模板图片
            cv2.imwrite('template.jpg', item_region)
            
            # 显示结果
            # cv2.imshow('Template', item_region)
            cv2.waitKey(1)
            
            print(f"模板已保存为 template.jpg，尺寸: {item_region.shape}")
            return True
            
        except Exception as e:
            print(f"提取模板过程中出错: {str(e)}")
            return False
    
    def detect_sample_item(self):
        """
        检测到与模板匹配的物体时画出边界框
        Returns:
            dict: 包含位置和置信度的字典，如果检测失败返回None
        """
        try:
            # 读取模板图像
            template = cv2.imread('template.jpg', cv2.IMREAD_GRAYSCALE)
            if template is None:
                print("错误：无法读取模板图像")
                return None

            # 获取摄像头图像
            frame = self.get_camera_image()

            # 转换为灰度图
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 创建SIFT对象
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(template, None)
            kp2, des2 = sift.detectAndCompute(gray, None)

            # BFMatcher进行特征点匹配
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # 选取优秀匹配点
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) > 4:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                h, w = template.shape
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)
                
                # 计算矩形框的左上角和右下角坐标
                x_coords = [int(p[0][0]) for p in dst]
                y_coords = [int(p[0][1]) for p in dst]
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                
                # 在原图上绘制矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                
                # 添加文本标签
                cv2.putText(frame, 'Matched Area', (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # 显示结果
                cv2.imshow('Detection Result', frame)
                cv2.waitKey(1)  # 等待1毫秒
                position_2d = (x1, y1, x2, y2)
                t1 = self.is_in_1(position_2d)
                x1_offset, y1_offset = self.is_in_2(position_2d)
                t2 = self.is_in_2(position_2d) 
                x1_offset, y1_offset = self.is_in_3(position_2d)
                t3 = self.is_in_3(position_2d)
                x1_offset, y1_offset = self.is_in_4(position_2d)
                t4 = self.is_in_4(position_2d)
                

                # 返回匹配区域的位置信息
                result = {
                    'position': position_2d,
                    'confidence': len(good) / len(matches) if len(matches) > 0 else 0,
                    'is_in_1': t1,
                    'is_in_2': t2,
                    'is_in_3': t3,
                    'is_in_4': t4
                }
                return result
            else:
                print("警告：未找到足够的匹配点")
                return None

        except Exception as e:
            print(f"检测过程中出错: {str(e)}")
            return None
    
    def detect_drinks(self, image, target_drinks):
        """
        检测指定饮料并返回3D坐标
        Args:
            image: 输入图像
            target_drinks: 需要检测的饮料列表
        Returns:
            list: 检测到的饮料位置列表
        """
        results = self.model.predict(source=image)
        detected_drinks = []

        for result in results:
            boxes = result.boxes
            # print(boxes)
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                # print(class_name)
                
                if class_name in target_drinks and confidence > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    position_2d = (x1, y1, x2, y2)
                    t1 = self.is_in_1(position_2d)
                    
                    detected_drink = {
                        'name': class_name,
                        'position_2d': position_2d,
                        'confidence': confidence,
                        'is_in_1': t1
                    }
                    # 递归转换所有np.float32类型
                    converted_drink = self.convert_np_floats(detected_drink)
                    detected_drinks.append(converted_drink)
        return detected_drinks

# def test():
#     # 测试代码
#     detector = VisionDetector()
#     try:
#         while True:
#             # 这里不需要再调用 detector.start_detection()，因为已经在 __init__ 中启动了线程
#             cv2.waitKey(1)
#     except KeyboardInterrupt:
#         print("程序终止")
#     finally:
#         cv2.destroyAllWindows()
#         detector.cap.release()

# def test1():
#     detector = VisionDetector()
#     detector.start_detection()

def test2():
    detector = VisionDetector()
    detector.start_detection()

if __name__ == "__main__":
    # test()
    # test1()

    
    # print("1")
    #     # 启动检测线程
    # detection_thread = threading.Thread(target=test1)
    # # detection_thread.daemon = True
    # detection_thread.start()
    # detection_thread.join()
    # print("2")
    test2()