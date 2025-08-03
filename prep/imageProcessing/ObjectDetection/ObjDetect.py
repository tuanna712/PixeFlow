import sys, base64
from PIL import Image
from io import BytesIO
from pathlib import Path

sys.path.append(str(Path('impl.ipynb').resolve().parents[3]))

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

class ObjectDetector:
    def __init__(self, co_detr_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = DetrImageProcessor.from_pretrained(co_detr_dir)
        self.model = DetrForObjectDetection.from_pretrained(co_detr_dir)
        self.model.to(self.device)
        self.returns= {}

    def detect(self, image_bs64):
        image_data = base64.b64decode(image_bs64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # Convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])

        # Let's only keep detections with score > 0.9
        self.results = self.processor.post_process_object_detection(outputs, 
                                                        target_sizes=target_sizes, 
                                                        threshold=0.9)[0]
        
        # Convert results to a more usable format
        self.returns["boxes"] = self.results["boxes"].tolist()
        self.returns["scores"] = self.results["scores"].tolist()
        self.returns["labels"] = [self.model.config.id2label[label.item()] for label in self.results["labels"]]

        return self.returns

    def visualize(self, image_bs64):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        image_data = base64.b64decode(image_bs64)
        image = Image.open(BytesIO(image_data)).convert("RGB")

        plt.figure(figsize=(12, 12))
        plt.imshow(image)
        ax = plt.gca()
        for score, label, box in zip(self.results["scores"], self.results["labels"], self.results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            rect = patches.Rectangle(
                (box[0], box[1]), box[2] - box[0], box[3] - box[1],
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(box[0], box[1], f"{self.model.config.id2label[label.item()]}: {score:.2f}",
                    bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')
        plt.axis('off')
        plt.show()