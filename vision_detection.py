import os
os.environ['YOLO_VERBOSE'] = str(False)
from ultralytics import YOLO
import cv2
import numpy as np
import torch
from rapidocr import RapidOCR

class VisionDetector:
    def __init__(self):
        self.model = YOLO('best.pt')  
        
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
                
                # 绘制0区域矩形框（蓝色）
                x1 = center_x - 120
                y1 = center_y + 180
                x2 = center_x + 120
                y2 = center_y - 200
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 5)  # 蓝色矩形框
                
                # 绘制1区域矩形框（蓝色）
                x11 =30
                y11 = 900
                x12 = 230
                y12 = 1400
                cv2.rectangle(frame, (x11, y11), (x12, y12), (255, 0, 0), 5)  # 蓝色矩形框
                
                # 绘制2区域矩形框（蓝色）
                x21 =30
                y21 = 1500
                x22 = 280
                y22 = 2100
                cv2.rectangle(frame, (x21, y21), (x22, y22), (255, 0, 0), 5)  # 蓝色矩形框
                
                # 绘制3区域矩形框（蓝色）
                x31 =890
                y31 = 150
                x32 = 1090
                y32 = 850
                cv2.rectangle(frame, (x31, y31), (x32, y32), (255, 0, 0), 5)  # 蓝色矩形框
                
                # 绘制4区域矩形框（蓝色）
                x41 =830
                y41 = 1550
                x42 = 970
                y42 = 2100
                cv2.rectangle(frame, (x41, y41), (x42, y42), (255, 0, 0), 5)  # 蓝色矩形框

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
    
    def recognize_shopping_list(self,image_path):
        # 初始化 RapidOCR
        ocr = RapidOCR()
        result = ocr(image_path)
        if not result:  # 现在这个检查只确保result不是空列表
            print("没有识别到任何内容")
            return []

        recognized_items = []  # 存储识别结果的数组
        if result is not None:  # 添加检查确保result不是None
            for line in result:
                if line:  # 确保line本身不为空
                    for element in line:
                        if isinstance(element, list) and len(element) > 1:
                            box = element[0]  # 包围盒坐标
                            text = element[1][0]  # 识别的文本
                            # 计算中心坐标
                            center_x = sum([point[0] for point in box]) / 4
                            center_y = sum([point[1] for point in box]) / 4
                            # 将文本和中心坐标存储为元组，并添加到数组中
                            recognized_items.append([text, (center_x, center_y)])
        return recognized_items
    
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
        # 计算目标物体的中心点
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        
        # 定义中心区域
        x11 =30
        y11 = 900
        x12 = 230
        y12 = 1400
        
        # # 计算相对于中心区域的偏移量
        # x_offset = 0
        # y_offset = 0
        
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
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        
        # 定义中心区域(待定)
        x21 =30
        y21 = 1500
        x22 = 280
        y22 = 2100
        
        if x21 < target_center_x < x22 and y21 < target_center_y < y22:
            return True
        else:
            return False

    
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
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        
        # 定义中心区域(待定)
        x31 =890
        y31 = 150
        x32 = 1090
        y32 = 850
        
        if x31 < target_center_x < x32 and y31 < target_center_y < y32:
            return True
        else:
            return False
    
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
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        
        # 定义中心区域(待定)
        x41 =830
        y41 = 1550
        x42 = 970
        y42 = 2100

        if x41 < target_center_x < x42 and y41 < target_center_y < y42:
            return True
        else:
            return False

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
                t2 = self.is_in_2(position_2d) 
                t3 = self.is_in_3(position_2d)
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
        检测指定饮料并返回坐标
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
                    t2 = self.is_in_2(position_2d)
                    t3 = self.is_in_3(position_2d)
                    t4 = self.is_in_4(position_2d)
                    
                    detected_drink = {
                        'name': class_name,
                        'position_2d': position_2d,
                        'confidence': confidence,
                        'is_in_1': t1,
                        'is_in_2': t2,
                        'is_in_3': t3,
                        'is_in_4': t4
                    }
                    # 递归转换所有np.float32类型
                    converted_drink = self.convert_np_floats(detected_drink)
                    detected_drinks.append(converted_drink)
        return detected_drinks
    
     

    def detect_labels(self, image, label):
        """
        检测指定标签并返回坐标
        Args:
            image: 输入图像
            label: 需要检测的标签
        Returns:
            list: 检测到的标签位置列表
        """
        ocr = RapidOCR()
        results = ocr(image, det_box=True ) 
        detected_labels = []

        for result in results:
            boxes = results
            # print(boxes)
            for box in boxes:
                confidence = float(result['scores'][0])
                class_name = result['txts']
                # print(class_name)
                
                if class_name in label and confidence > 0.2:
                    x1, y1, x2, y2 = map(int, result['boxes'])
                    position_2d = (x1, y1, x2, y2)
                    t3 = self.is_in_3(position_2d)
                    
                    detected_label = {
                        'name': class_name,
                        'position_2d': position_2d,
                        'confidence': confidence,
                        'is_in_3': t3
                    }
                    # 递归转换所有np.float32类型
                    converted_label = self.convert_np_floats(detected_label)
                    detected_labels.append(converted_label)
        return detected_labels

def test2():
    detector = VisionDetector()
    detector.start_detection()

if __name__ == "__main__":
    test2()