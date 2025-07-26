from dataclasses import dataclass
from typing import Any, Optional, Dict, List

dataclass
class EmbeddingRecord:
    id: int | None
    video_id: str
    keyframe_id: str
    embeddings: Dict[str, List[float]]  # Dictionary chứa image, text
    extra_info: Dict[str, Any]
    score: float = 0.0

@dataclass
class VideoRecord:
    video_id: str
    video_name: str
    file_name: str
    file_path: str
    file_type: str
    file_date: str
    duration_sec: float
    json_path: str
    keyframe_dir: str

@dataclass
class SceneRecord:
    scene_id: Optional[int]  # None for new/insertion, int for fetch/update
    video_id: str
    start_frame: int
    end_frame: int

@dataclass
class KeyframeRecord:
    keyframe_id: Optional[int]
    video_id: str
    scene_id: int
    frame_idx: int
    file_path: str
    timestamp: float
    embedding_id: str
    metadata_json: Any  # dict or str depending on your use