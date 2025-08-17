import numpy as np
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
from io import BytesIO

from paddleocr import PaddleOCR

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


class OCR:
    def __init__(self):
        self.ocr_model = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="gpu:0" if torch.cuda.is_available() else "cpu"
        )
        self.config = Cfg.load_config_from_name("vgg_transformer")
        self.config["cnn"]["pretrained"] = False
        self.config["device"] = "cuda:0"
        self.viet_ocr = Predictor(self.config)

    def detect_text_box(self, image):
        result = self.ocr_model.predict(input=image.copy())
        boxes = []
        for i in result[0]["rec_boxes"]:
            boxes.append(i)
        return boxes

    def get_crop_image(self, boxes, image):
        crop_images = []

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            crop_img = image[y_min : y_max + 1, x_min : x_max + 1]
            crop_images.append(crop_img)

        return crop_images

    def recognize_text(self, crop_images):
        list_text = []

        for crop_image in crop_images:
            img = Image.fromarray(crop_image)
            result = self.viet_ocr.predict(img)
            list_text.append(result)

        return list_text

    def run_ocr(self, image):
        boxes = self.detect_text_box(image)
        crop_images = self.get_crop_image(boxes.copy(), image.copy())
        list_text = self.recognize_text(crop_images)
        return list_text
