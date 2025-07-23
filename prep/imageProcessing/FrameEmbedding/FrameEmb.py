import sys, base64
from PIL import Image
from io import BytesIO
from pathlib import Path

sys.path.append(str(Path('impl.ipynb').resolve().parents[3]))

import torch
from transformers import CLIPProcessor, CLIPModel



class ClipEncoder:
    def __init__(self, clip_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(clip_dir)
        self.model = CLIPModel.from_pretrained(clip_dir)
        self.model.to(self.device)

    def text_encode(self, text):
        input = self.processor(text=text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        encoded_input = self.model.get_text_features(**input)
        return encoded_input.detach().cpu().numpy().squeeze(0)
    
        
    def image_encode(self, image_bs64):
        image_data = base64.b64decode(image_bs64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        input = self.processor(images=image, return_tensors="pt", padding=True).to(self.device)
        encoded_input = self.model.get_image_features(**input)
        return encoded_input.detach().cpu().numpy().squeeze(0)
            