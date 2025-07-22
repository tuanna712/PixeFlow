## Co-DETR model from Hugging Face

Execute the following code to download the model:<br>
From repo: [Hugging Face DETR for Object Detection](https://huggingface.co/facebook/detr-resnet-101)<br>
Publication: [(2020) End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)

```python
import os, shutil
sys.path.append(str(Path('image_proc.ipynb').resolve().parents[3]))
from transformers import DetrImageProcessor, DetrForObjectDetection
from prep.params import CO_DETR_DIR

# Define temp folder
TEMP_DIR = "./.co_detr_model"
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Get Processor
processor = DetrImageProcessor.from_pretrained(
    "facebook/detr-resnet-101", 
    cache_dir=TEMP_DIR,
    revision="no_timm"
)

# Get model
model = DetrForObjectDetection.from_pretrained(
    "facebook/detr-resnet-101", 
    cache_dir=TEMP_DIR, 
    revision="no_timm"
)
# Remove temp_dir
shutil.rmtree(TEMP_DIR, ignore_errors=True)

# Save model to local for later uses
processor.save_pretrained(CO_DETR_DIR)
model.save_pretrained(CO_DETR_DIR)

# Load model from local
processor = DetrImageProcessor.from_pretrained(CO_DETR_DIR)
model = DetrForObjectDetection.from_pretrained(CO_DETR_DIR)
```


Code for implementation and visualization

```python
# Define Image
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image_1 = Image.open(requests.get(url, stream=True).raw)
path = "local_image.jpg"
image_2 = Image.open(path)

# Load image to processor
inputs = processor(images=image_2, return_tensors="pt").to(device)
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image_2.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )

# Vizualize the results
import matplotlib.pyplot as plt
import matplotlib.patches as patches

plt.figure(figsize=(12, 12))
plt.imshow(image_2)
ax = plt.gca()
for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    rect = patches.Rectangle(
        (box[0], box[1]), box[2] - box[0], box[3] - box[1],
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    plt.text(box[0], box[1], f"{model.config.id2label[label.item()]}: {score:.2f}",
             bbox=dict(facecolor='yellow', alpha=0.5), fontsize=12, color='black')
plt.axis('off')
plt.show()
```
