import os, shutil
import sys, base64
from PIL import Image
from io import BytesIO
from pathlib import Path

sys.path.append(str(Path('impl.ipynb').resolve().parents[3]))

import torch
from transformers import AutoProcessor, BlipForConditionalGeneration

class BlipCaptioner:
    def __init__(self, blip_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(blip_dir)
        self.model = BlipForConditionalGeneration.from_pretrained(blip_dir)
        self.model.to(self.device)

    def caption(self, image_bs64):
        image_data = base64.b64decode(image_bs64)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, 
                                            max_length=50,
                                            do_sample=True,
                                            top_k=50,
                                            top_p=0.9,
                                            temperature=0.5,
                                            num_beams=1,            # Set to 1 for greedy or sampling. Use >1 for beam search (usually without do_sample=True).
                                            repetition_penalty=1.2, # Penalize repeating n-grams
                                        )

        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        caption = caption.strip()
        if caption.endswith('.'):
            caption = caption[:-1]
        return caption