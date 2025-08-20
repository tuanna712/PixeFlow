from transformers import BeitFeatureExtractor, BeitModel
from PIL import Image
import requests
import torch

def embedding(image_path, size='large'):
    # 1. Pre-trained BEiT model
    if size == 'large':
        model_name = 'microsoft/beit-large-patch16-224'
    else:
        model_name = 'microsoft/beit-base-patch16-224'
        
    feature_extractor = BeitFeatureExtractor.from_pretrained(model_name)
    model = BeitModel.from_pretrained(model_name)
    
    # 2. Load an image
    image = Image.open(image_path).convert('RGB')
    
    # 3. Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    # 4. Get the model output
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    embeddings = outputs.pooler_output
    
    return embeddings