from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class EmbeddingRecord:
    id: int | None
    video_id: str
    keyframe_id: str
    embeddings: Dict[str, List[float]]  # Dictionary chứa image, text
    extra_info: Dict[str, Any]
    score: float = 0.0