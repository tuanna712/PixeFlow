import json
import os

from mmdet.apis import init_detector, inference_detector
from mmdet.core import DatasetEnum

def extract_image_metadata(
    img_path,
    config_file='Co-DETR/checkpoint/co_deformable_detr_swin_base_3x_coco.py',
    checkpoint_file='Co-DETR/checkpoint/co_deformable_detr_swin_base_3x_coco.pth',
    device='cuda:0',
    score_thr=0.3
):
    # Load model
    model = init_detector(config_file, checkpoint_file, DatasetEnum.COCO, device=device)

    # Inference
    results = inference_detector(model, img_path)

    # Load class labels
    if hasattr(model, 'CLASSES'):
        class_names = model.CLASSES
    elif hasattr(model.dataset_meta, 'classes'):
        class_names = model.dataset_meta['classes']
    else:
        # Default to COCO if metadata is missing
        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    # Prepare metadata
    metadata = {}
    metadata['image_id'] = os.path.abspath(img_path)

    for class_id, class_detections in enumerate(results):
        if len(class_detections) == 0:
            continue

        # Filter by score threshold
        filtered = class_detections[class_detections[:, -1] >= score_thr]
        if filtered.shape[0] == 0:
            continue

        class_name = class_names[class_id]

        metadata[class_name] = {
            'count': filtered.shape[0],
            'boxes': filtered.tolist()  # each box is [x1, y1, x2, y2, score]
        }

    return metadata

if __name__ == '__main__':
    img_path = 'demo/demo.jpg'  # Path to your image
    metadata = extract_image_metadata(img_path)
    with open('demo/demo.json', 'w') as f:
        json.dump(metadata, f, indent=4)
    print("Extracted Metadata:")
    for class_name, data in metadata.items():
        if class_name != 'image_id':
            print(f"{class_name}: {data['count']} detections")
            for box in data['boxes']:
                print(f"  Box: {box}")