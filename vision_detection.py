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
        self.ocr = RapidOCR()
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
        
        self.label = ["锐澳", "东鹏", "茶Ⅱ", "加多宝"]#["锐澳水蜜桃", "东鹏", "茶π柠檬红茶", "加多宝"]

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
                
                # 绘制0区域矩形框（蓝色）
                x01 = 600
                y01 = 1030
                x02 = 840
                y02 = 1460
                cv2.rectangle(frame, (x01, y01), (x02, y02), (255, 0, 0), 3)  # 蓝色矩形框
                
                # 绘制1区域矩形框（蓝色）
                x11 =30
                y11 = 950
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
                x31 =780
                y31 = 150
                x32 = 1090
                y32 = 650
                cv2.rectangle(frame, (x31, y31), (x32, y32), (255, 0, 0), 5)  # 蓝色矩形框
                
                # 绘制4区域矩形框（蓝色）
                x41 =850
                y41 = 1550
                x42 = 1000
                y42 = 2000
                cv2.rectangle(frame, (x41, y41), (x42, y42), (255, 0, 0), 5)  # 蓝色矩形框

                # 绘制5区域矩形框（蓝色）
                x51 =1000
                y51 = 650
                x52 = 1160
                y52 = 850
                cv2.rectangle(frame, (x51, y51), (x52, y52), (255, 0, 0), 5)  # 蓝色矩形框
                cv2.line(frame, (0,y51), (width,y51), (255, 0, 0), 5)  # 蓝色水平线
                cv2.line(frame, (0,y52), (width,y52), (255, 0, 0), 5)  # 蓝色水平线

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
    
    # def is_in_0(self, bbox):
    #     """
    #     判断目标是否在0区域
    #     Args:
    #         bbox: 边界框坐标 (x1, y1, x2, y2)
    #     Returns:
    #         bool: 是否在标签区域
    #     """
    #     x1, y1, x2, y2 = bbox
    #     target_center_x = (x1 + x2) / 2
    #     target_center_y = (y1 + y2) / 2
        
    #     # 定义中心区域(待定)
    #     x01 = 600
    #     y01 = 1080
    #     x02 = 840
    #     y02 = 1460

    #     if x01 < target_center_x < x02 and y01 < target_center_y < y02:
    #         return True
    #     else:
    #         return False
    
    
    
    def is_in_1(self, bbox):
        """
        判断目标是否在1区域
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
        Returns:
            bool: 是否在标签区域
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
        y11 = 950
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
        判断目标是否在2区域
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
        Returns:
            bool: 是否在标签区域
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
        判断目标是否在3区域
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
        Returns:
            bool: 是否在标签区域
        """
        x1, y1, x2, y2 = bbox
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        
        # 定义中心区域(待定)
        x31 =780
        y31 = 150
        x32 = 1090
        y32 = 650
        
        if x31 < target_center_x < x32 and y31 < target_center_y < y32:
            return True
        else:
            return False
    
    def is_in_4(self, bbox):
        """
        判断目标是否在4区域
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
        Returns:
            bool: 是否在标签区域
        """
        x1, y1, x2, y2 = bbox
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        
        # 定义中心区域(待定)
        x41 =850
        y41 = 1550
        x42 = 1000
        y42 = 2000

        if x41 < target_center_x < x42 and y41 < target_center_y < y42:
            return True
        else:
            return False
        
    # def is_in_5(self, bbox):
    #     """
    #     判断目标是否在5区域
    #     Args:
    #         bbox: 边界框坐标 (x1, y1, x2, y2)
    #     Returns:
    #         bool: 是否在标签区域
    #     """
    #     x1, y1, x2, y2 = bbox
    #     target_center_x = (x1 + x2) / 2
    #     target_center_y = (y1 + y2) / 2
        
    #     # 定义中心区域(待定)
    #     x51 =930 
    #     y51 = 700
    #     x52 = 1230
    #     y52 = 900

    #     if x51 < target_center_x < x52 and y51 < target_center_y < y52:
    #         return True
    #     else:
    #         return False

    def for_label_area(self, bbox):
        """
        判断目标在标签区域的状态
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
        Returns:
            result: 是否在标签区域
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 定义中心区域(待定)
        x51 =1000 
        y51 = 650
        x52 = 1160
        y52 = 850

        tx = False
        ty = False

        # 计算水平偏移
        if center_x < x51:
            x_offset = center_x - x51  # 负值，表示在左侧
        elif center_x > x52:
            x_offset = center_x - x52  # 正值，表示在右侧
        elif x51 < center_x < x52:
            x_offset = 0
            tx = True

        if  y51 < center_y < y52:
            ty = True #在标签检测区域
        else:
            ty = False

        print("x_offset:", x_offset)
        
        result = {
                    'tx': tx,
                    'ty': ty,
                    'x_offset': x_offset
                }
        return  result
    
    def detect_sample(self, image):
        """
        提取图像中心的物品并保存为模板图片
        Returns:
            bool: 是否成功保存模板
        """
        x01 = 600
        y01 = 1030
        x02 = 840
        y02 = 1460

        x01 = x01 + 20
        y01 = y01 + 20
        x02 = x02 - 20
        y02 = y02 - 20
        
        # 提取中心区域
        center_region = image[y01:y02, x01:x02]
        
        # 转换为灰度图
        gray = cv2.cvtColor(center_region, cv2.COLOR_BGR2GRAY)
        
        # 应用双边滤波
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        cv2.imshow("Filtered", filtered)
        
        # 应用自适应阈值处理
        thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        cv2.imshow("Threshold", thresh)

        # Canny边缘检测
        edges = cv2.Canny(thresh, 150, 200)  # 100和200是阈值
        cv2.imshow("Edges", edges)
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("警告：未在中心区域找到物品")
            return False
        
        # 找到最大的轮廓
        max_contour = max(contours, key=cv2.contourArea)
        
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 在原图上绘制边界框
        cv2.rectangle(center_region, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Center Region", center_region)
        
        # 提取物品区域
        item_region = center_region[y:y+h, x:x+w]
        
        # 保存模板图片
        cv2.imwrite('template.jpg', item_region)
        
        cv2.waitKey(1)
        
        print(f"模板已保存为 template.jpg，尺寸: {item_region.shape}")
        return True
        
    
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
                # cv2.imshow('Detection Result', frame)
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
                    # t5 = self.is_in_5(position_2d)
                    
                    detected_drink = {
                        'name': class_name,
                        'position_2d': position_2d,
                        'confidence': confidence,
                        'is_in_1': t1,
                        'is_in_2': t2,
                        'is_in_3': t3,
                        'is_in_4': t4,
                        # 'is_in_5': t5
                    }
                    # 递归转换所有np.float32类型
                    converted_drink = self.convert_np_floats(detected_drink)
                    detected_drinks.append(converted_drink)
        return detected_drinks
    def label_correct(self, label):
        """
        设置需要更改的标签
        Args:
            label: 需要更改的标签
        """
        if label.startswith("茶"):
            return "茶Ⅱ"
        else:
            return label  # 如果没有匹配的标签，返回原始标签

    def detect_labels(self, image, label_list, label_name):
        """
        检测指定标签并返回坐标
        Args:
            image: 输入图像
            label: 需要检测的标签
        Returns:
            list: 检测到的标签位置列表
        """
        results = self.ocr(image) 
        detected_labels = []
        if results:
        # 遍历识别结果
            for box, txt, score in zip(results.boxes, results.txts, results.scores):
                # print(f"{box}Recognized text: {txt}, Confidence: {score:.2f}")
                confidence = float(score)
                class_name = txt
                if  confidence > 0.2 and self.label_correct(class_name) == label_name:
                    x1, y1, x2, y2 = map(int, [box[0][0], box[0][1], box[2][0], box[2][1]])
                    position_2d = (x1, y1, x2, y2)
                    label_area = self.for_label_area(position_2d)
                    detected_label = {
                        'name': class_name,
                        'position_2d': position_2d,
                        'confidence': confidence,
                        'tx': label_area['tx'],
                        'ty': label_area['ty'],
                        'x_offset': label_area['x_offset']
                    }
                    # 递归转换所有np.float32类型
                    converted_label = self.convert_np_floats(detected_label)
                    print(f"111:{converted_label}")
                    detected_labels.append(converted_label)
        return detected_labels



def test2():
    detector = VisionDetector()
    while True:
        image = detector.get_camera_image()
        detector.detect_sample(image)

if __name__ == "__main__":
    test2()