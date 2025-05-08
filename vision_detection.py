from ultralytics import YOLO
import cv2
import numpy as np
import torch

class VisionDetector:
    def __init__(self):
        self.model = YOLO('best.pt', verbose=False)  
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
    
    def start_detection(self):
        """
        开始实时检测并显示窗口
        """
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 蓝色矩形框
            
            # 运行模型检测
            results = self.model(frame)
            
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
        
        # 释放资源
        cv2.destroyAllWindows()
    
    def get_camera_image(self):
        """
        获取摄像头图像并实时显示（最高分辨率）
        Returns:
            numpy.ndarray: 图像数据
        """
        try:
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("无法读取图像")
                
            
                # cv2.imshow('Camera', frame)
                
                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            return frame
        
        finally:
            cv2.destroyAllWindows()
    
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
    
    def is_in_center(self, bbox, image):
        """
        计算边界框相对于中心区域的偏移量
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
            image: 输入图像
        Returns:
            tuple: (x_offset, y_offset) 偏移量
                  x_offset: 正值表示目标在中心区域右侧，负值表示目标在中心区域左侧
                  y_offset: 正值表示目标在中心区域下方，负值表示目标在中心区域上方
        """
        # 获取图像中心点
        height, width = image.shape[:2]
        center_x = width // 2
        center_y = height // 2
        
        # 定义中心区域
        x1 = center_x - 120
        y1 = center_y + 180
        x2 = center_x + 120
        y2 = center_y - 200
        
        # 计算目标物体的中心点
        target_center_x = (bbox[0] + bbox[2]) / 2
        target_center_y = (bbox[1] + bbox[3]) / 2
        
        # 计算相对于中心区域的偏移量
        x_offset = 0
        y_offset = 0
        
        # 计算水平偏移
        if target_center_x < x1:
            x_offset = target_center_x - x1  # 负值，表示在左侧
        elif target_center_x > x2:
            x_offset = target_center_x - x2  # 正值，表示在右侧
            
        # 计算垂直偏移
        if target_center_y < y2:
            y_offset = target_center_y - y2  # 负值，表示在上方
        elif target_center_y > y1:
            y_offset = target_center_y - y1  # 正值，表示在下方
        
        return (x_offset, y_offset)
    
    def is_in_Xcenter(self, bbox, image):
        """
        计算边界框相对于X轴中心区域的偏移量
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
            image: 输入图像
        Returns:
            float: 偏移量，正值表示目标在中心区域右侧，负值表示目标在中心区域左侧
        """
        # 获取图像中心点
        width = image.shape[:1]
        center_x = width // 2
        
        # 定义中心区域
        x1 = center_x - 120
        x2 = center_x + 120
        
        # 计算目标物体的中心点
        target_center_x = (bbox[0] + bbox[2]) / 2
        
        # 计算相对于中心区域的偏移量
        x_offset = 0
        
        # 计算水平偏移
        if target_center_x < x1:
            x_offset = target_center_x - x1  # 负值，表示在左侧
        elif target_center_x > x2:
            x_offset = target_center_x - x2  # 正值，表示在右侧
            
        return x_offset

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
            ret, frame = self.cap.read()
            if not ret:
                print("错误：无法读取摄像头图像")
                return None

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
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # 在原图上绘制矩形框
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                
                # 添加文本标签
                cv2.putText(frame, 'Matched Area', (x_min, y_min-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # 显示结果
                # cv2.imshow('Detection Result', frame)
                cv2.waitKey(1)  # 等待1毫秒

                # 返回匹配区域的位置信息
                return {
                    'position': (x_min, y_min, x_max, y_max),
                    'confidence': len(good) / len(matches) if len(matches) > 0 else 0
                }
            else:
                print("警告：未找到足够的匹配点")
                return None

        except Exception as e:
            print(f"检测过程中出错: {str(e)}")
            return None
    
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

    def detect_drinks(self, image, target_drinks):
        """
        检测指定饮料并返回3D坐标
        Args:
            image: 输入图像
            target_drinks: 需要检测的饮料列表
        Returns:
            list: 检测到的饮料位置列表
        """
        image_tensor = torch.from_numpy(image).to(self.device)
        results = self.model(image_tensor)
        detected_drinks = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                if class_name in target_drinks and confidence > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].gpu(0).numpy()
                    position_2d = (x1, y1, x2, y2)
                    position_3d = self.convert_to_3d_coordinates(position_2d)
                    
                    detected_drinks.append({
                        'name': class_name,
                        'position_2d': position_2d,
                        'position_3d': position_3d,
                        'confidence': confidence,
                        'is_in_center': self.is_in_center(position_2d, image),
                        'is_in_Xcenter': self.is_in_Xcenter(position_2d, image)
                    })
        
        return detected_drinks

def test():
    # 测试代码
    detector = VisionDetector()
    detector.start_detection()  # 开始实时检测
    
    # 获取图像
    # image = detector.get_camera_image()
    
    # # 检测样品
    # sample_item = detector.detect_sample(image)
    # print(f"检测到的样品: {sample_item}")

    # # 检测饮料
    # target_drinks = ["yykx", "wz", "bs", "yld"]
    # detected_drinks = detector.detect_drinks(image, target_drinks)
    # print(f"检测到的饮料: {detected_drinks}")

if __name__ == "__main__":
    test()