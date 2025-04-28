from ultralytics import YOLO
import cv2
import numpy as np

class VisionDetector:
    def __init__(self):
        self.model = YOLO('best.pt')  # 使用预训练模型，您也可以使用自己训练的模型
        self.image_center_threshold = 50  # 图像中心区域的阈值（像素）
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) #闭运算卷积核
        self.template_path = "5.jpg"
        
        # 打开摄像头
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("无法打开摄像头")
        
        # 获取摄像头支持的最高分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)  # 设置一个很大的值
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
        
        # 获取实际设置的分辨率
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"摄像头分辨率: {actual_width}x{actual_height}")

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
                
                # 应用双边滤波
                filtered_frame = cv2.bilateralFilter(frame, 9, 75, 75)
                # 应用闭运算
                filtered_frame = cv2.morphologyEx(filtered_frame, cv2.MORPH_CLOSE, self.kernel)
                
                cv2.imshow('Camera', filtered_frame)
                
                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            return filtered_frame
        
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
    
    def is_in_center(self, bbox, image_center):
        """
        判断边界框是否在图像中心区域
        Args:
            bbox: 边界框坐标 (x1, y1, x2, y2)
            image_center: 图像中心坐标 (center_x, center_y)
        Returns:
            bool: 是否在中心区域
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        return abs(center_x - image_center[0]) < self.image_center_threshold
    
    def detect_item(self):
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
            cv2.imshow('Template', item_region)
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
                cv2.imshow('Detection Result', frame)
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
        results = self.model(image)
        detected_drinks = []
        image_center = self.get_image_center(image)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                if class_name in target_drinks and confidence > 0.5:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    position_2d = (x1, y1, x2, y2)
                    position_3d = self.convert_to_3d_coordinates(position_2d)
                    
                    detected_drinks.append({
                        'name': class_name,
                        'position_2d': position_2d,
                        'position_3d': position_3d,
                        'confidence': confidence,
                        'is_in_center': self.is_in_center(position_2d, image_center)
                    })
        
        return detected_drinks
    
    
    def test_model(self):
        """
        测试模型检测效果，实时显示检测结果
        """
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    raise Exception("无法读取图像")
                # 应用双边滤波
                filtered_frame = cv2.medianBlur(frame, 5)
                # # 应用闭运算
                # filtered_frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, self.kernel)
                
                
                # 运行模型检测
                results = self.model(filtered_frame)
                
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
                        cv2.rectangle(filtered_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        # 添加标签文本
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(filtered_frame, label, (int(x1), int(y1-10)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow('Model Test', filtered_frame)
                
                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    
def test():
    # 测试代码
    detector = VisionDetector()
    
    # 获取图像
    image = detector.get_camera_image()
    
    # 检测样品
    sample_item = detector.detect_item(image)
    print(f"检测到的样品: {sample_item}")
    
    # # # 检测饮料
    # target_drinks = ["yykx", "wz", "bs", "yld"]
    # detected_drinks = detector.detect_drinks(image, target_drinks)
    # print(f"检测到的饮料: {detected_drinks}")

    detector.test_model()

if __name__ == "__main__":
    test() 