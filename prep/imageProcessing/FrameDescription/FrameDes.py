import sys
from pathlib import Path

sys.path.append(str(Path('impl.ipynb').resolve().parents[3]))

from data.models.LLaVA.llava.mm_utils import get_model_name_from_path
from data.models.LLaVA.llava.eval.run_llava import eval_model

class FrameDescription:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = get_model_name_from_path(model_path)
    
    def get_description(self,
                        image_bs64: str,
                        prompt: str = "Act as an expert writer who can analyze an imagery and explain it in a descriptive way, using as much detail as possible from the image.The content should be 300 words minimum."
                        ) -> str:
        args = type('Args', (), {
            "model_path": self.model_path,
            "model_base": None,
            "model_name": self.model_name,
            "query": prompt,
            "conv_mode": None,
            "image_file": image_bs64,
            "sep": ",",
            "temperature": 0.1,
            "top_p": 0.9,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

        responses = eval_model(args)
        description = ""
        for response in responses: 
            description += response
        return description
