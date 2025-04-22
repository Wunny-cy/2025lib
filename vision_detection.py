from ultralytics import YOLO
import cv2

class VisionDetector:
    def __init__(self):
        # 加载YOLOv8模型
        self.model = YOLO('best.pt')  # 使用预训练模型，您也可以使用自己训练的模型
        self.image_center_threshold = 50  # 图像中心区域的阈值（像素）
        
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
    
    def detect_sample_item(self, image):
        """
        检测高台小方桌上的样品物品
        Args:
            image: 输入图像
        Returns:
            str: 检测到的物品名称
        """
        # 运行YOLO11检测
        results = self.model(image)
        
        # 获取检测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取置信度
                confidence = float(box.conf[0])
                # 获取类别
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # 如果置信度大于阈值，返回检测到的物品
                if confidence > 0.7:  # 可以调整这个阈值
                    return class_name
        
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
    
    def get_camera_image(self):
        """
        获取摄像头图像并实时显示（最高分辨率）
        Returns:
            numpy.ndarray: 图像数据
        """
        # 打开摄像头
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("无法打开摄像头")
        
        try:
            # 获取摄像头支持的最高分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)  # 设置一个很大的值
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
            
            # 获取实际设置的分辨率
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"摄像头分辨率: {actual_width}x{actual_height}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    raise Exception("无法读取图像")
                
                # 显示图像
                cv2.imshow('Camera', frame)
                
                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            return frame
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def test_model(self):
        """
        测试模型检测效果，实时显示检测结果
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("无法打开摄像头")
        
        try:
            # 设置最高分辨率
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
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
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 显示图像
                cv2.imshow('Model Test', frame)
                
                # 按'q'键退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

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

def main():
    # 测试代码
    detector = VisionDetector()
    
    # 获取图像
    # image = detector.get_camera_image()
    
    # # 检测样品
    # sample_item = detector.detect_sample_item(image)
    # print(f"检测到的样品: {sample_item}")
    
    # # 检测饮料
    # target_drinks = ["yykx", "wz", "bs", "yld"]
    # detected_drinks = detector.detect_drinks(image, target_drinks)
    # print(f"检测到的饮料: {detected_drinks}")

    detector.test_model()

if __name__ == "__main__":
    main() 