import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import numpy as np

class ObjectDetector:
    def __init__(self, model_path="yolov13x.pt"):
        """
        Khởi tạo ObjectDetector với YOLOv13
        
        Args:
            model_path (str): Đường dẫn đến file model YOLOv13 (.pt)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load YOLOv12 model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Lưu kết quả detection
        self.results = None
        self.returns = {}
        
        # COCO class names (YOLOv12 thường được train trên COCO dataset)
        self.class_names = self.model.names

    def detect(self, image, confidence_threshold=0.5):
        """
        Phát hiện object trong hình ảnh
        
        Args:
            image: PIL Image hoặc numpy array hoặc đường dẫn đến file ảnh
            confidence_threshold (float): Ngưỡng confidence để filter detection
            
        Returns:
            dict: Dictionary với format:
            {
                "object_name_1": {
                    "count": số_lượng_object,
                    "detections": [
                        {
                            "score": confidence_score,
                            "bbox": [x1, y1, x2, y2]
                        },
                        ...
                    ]
                },
                "object_name_2": {...},
                ...
            }
        """
        # Đảm bảo image là PIL Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Image phải là PIL Image, numpy array, hoặc đường dẫn file")
        
        # Chạy detection với YOLOv13
        self.results = self.model(image, conf=confidence_threshold, device=self.device, verbose=False)
        
        # Lấy kết quả từ detection đầu tiên (vì chỉ có 1 ảnh)
        result = self.results[0]
        
        # Khởi tạo dictionary để group theo object class
        detected_objects = {}
        
        if result.boxes is not None:
            # YOLOv12 trả về boxes ở format xyxy
            boxes = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            class_indices = result.boxes.cls.cpu().tolist()
            
            # Group detections theo class name
            for box, score, class_idx in zip(boxes, scores, class_indices):
                class_name = self.class_names[int(class_idx)]
                
                # Nếu class chưa có trong detected_objects, khởi tạo
                if class_name not in detected_objects:
                    detected_objects[class_name] = {
                        "count": 0,
                        "detections": []
                    }
                
                # Thêm detection vào class tương ứng
                detected_objects[class_name]["detections"].append({
                    "score": score,
                    "bbox": box
                })
                
                # Cập nhật count
                detected_objects[class_name]["count"] += 1
        
        self.returns = detected_objects
        return self.returns

    def visualize(self, image, confidence_threshold=0.5, figsize=(12, 12)):
        """
        Visualize kết quả detection trên hình ảnh
        
        Args:
            image: PIL Image hoặc numpy array hoặc đường dẫn đến file ảnh
            confidence_threshold (float): Ngưỡng confidence để filter detection
            figsize (tuple): Kích thước figure để hiển thị
        """
        # Đảm bảo image là PIL Image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("Image phải là PIL Image, numpy array, hoặc đường dẫn file")
        
        # Nếu chưa có kết quả detection, chạy detect trước
        if self.results is None:
            self.detect(image, confidence_threshold)
        
        # Tạo figure để hiển thị
        plt.figure(figsize=figsize)
        plt.imshow(image)
        ax = plt.gca()
        
        # Định nghĩa colors cho các class khác nhau
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
        color_map = {}
        color_idx = 0
        
        total_objects = 0
        
        # Vẽ bounding boxes và labels cho từng object class
        for object_name, object_data in self.returns.items():
            # Assign color cho class này
            if object_name not in color_map:
                color_map[object_name] = colors[color_idx % len(colors)]
                color_idx += 1
            
            color = color_map[object_name]
            total_objects += object_data["count"]
            
            # Vẽ từng detection của class này
            for detection in object_data["detections"]:
                score = detection["score"]
                x1, y1, x2, y2 = detection["bbox"]
                width = x2 - x1
                height = y2 - y1
                
                # Vẽ rectangle
                rect = patches.Rectangle(
                    (x1, y1), width, height,
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax.add_patch(rect)
                
                # Thêm text label và confidence score
                plt.text(x1, y1 - 5, f"{object_name}: {score:.2f}",
                        bbox=dict(facecolor=color, alpha=0.7), 
                        fontsize=10, color='white')
        
        plt.axis('off')
        plt.title(f"YOLOv12 Object Detection - {total_objects} objects detected")
        plt.tight_layout()
        plt.show()

    def get_detection_summary(self):
        """
        Trả về tóm tắt kết quả detection
        
        Returns:
            dict: Summary của detection results
        """
        if self.returns:
            total_objects = sum(obj_data["count"] for obj_data in self.returns.values())
            
            # Tính average confidence cho từng class
            class_avg_confidence = {}
            for object_name, object_data in self.returns.items():
                scores = [det["score"] for det in object_data["detections"]]
                class_avg_confidence[object_name] = sum(scores) / len(scores) if scores else 0
            
            return {
                "total_objects": total_objects,
                "unique_classes": len(self.returns.keys()),
                "class_counts": {obj_name: obj_data["count"] for obj_name, obj_data in self.returns.items()},
                "class_average_confidence": class_avg_confidence
            }
        return {"message": "Chưa có kết quả detection"}

    def print_detection_results(self):
        """
        In kết quả detection theo format dễ đọc
        """
        if not self.returns:
            print("Chưa có kết quả detection")
            return
        
        print("=== DETECTION RESULTS ===")
        total_objects = sum(obj_data["count"] for obj_data in self.returns.values())
        print(f"Total objects detected: {total_objects}")
        print(f"Unique classes: {len(self.returns.keys())}")
        print("-" * 30)
        
        for object_name, object_data in self.returns.items():
            print(f"\n{object_name.upper()}:")
            print(f"  Count: {object_data['count']}")
            print(f"  Detections:")
            
            for i, detection in enumerate(object_data["detections"], 1):
                score = detection["score"]
                bbox = detection["bbox"]
                print(f"    {i}. Score: {score:.3f}, BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")