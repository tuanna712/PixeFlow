# DETRs with Collaborative Hybrid Assignments Training (CO-DETR) for object detection

This guide provides instructions setting up Co-Deformable-DETR model with Swin-Base backbone. This one suitable for low-end GPUs with 4GB of GPU memory. All available Co-DETR models can be found at [Sense-X/Co-DETR](https://github.com/Sense-X/Co-DETR)

## Installation:
1. Install `mmcv` as a dependency in following [official instruction](https://github.com/open-mmlab/mmcv/tree/v1.5.0#installation). `mmcv` between ``1.3.17`` and ``1.7.0`` is recommended. The installation might take some time

2. Move to the `model` directory:
   ```bash
   cd ../../../model
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/Sense-X/Co-DETR.git
    ```
4. Install the `mmdet` package from the cloned repository:
   ```bash
   cd Co-DETR
   pip install -e .
   ``` 
5. Download whole `checkpoint` directory from [here](https://drive.google.com/drive/folders/1CAQ9sjxCT5e5wzpsHKGlqgylM504HcV7?usp=sharing), put it in `model/Co-DETR` directory

6. Run  `prep/imageProcessing/object_detect/visual_swin_base.py` to confirm the installation:
  

Note: If your `mmcv` is not in recommended version, `mmdet` might not happy. You can try to find `model/Co-DETR/mmdet/__init__.py` and edit these lines to match your version:
   ```python
mmcv_minimum_version = '1.3.17'
mmcv_maximum_version = '1.7.0'
   ```
