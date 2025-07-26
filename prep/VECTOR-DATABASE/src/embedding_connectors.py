from typing import List, Dict, Any
from pathlib import Path
import sys
import numpy as np
from loguru import logger

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models import EmbeddingRecord
from repositories.qdrant_repository import QdrantEmbeddingRepository

def create_embedding_record(
    video_id: str,
    keyframe_id: str,
    image_embedding: List[float] | None = None,
    text_embedding: List[float] | None = None,
    extra_info: Dict[str, Any] | None = None
) -> EmbeddingRecord:
    """Tạo EmbeddingRecord từ các vector embedding đã có."""
    embeddings = {}
    if image_embedding is not None:
        embeddings["image"] = image_embedding
    if text_embedding is not None:
        embeddings["text"] = text_embedding
    
    return EmbeddingRecord(
        id=None,
        video_id=video_id,
        keyframe_id=keyframe_id,
        embeddings=embeddings,
        extra_info=extra_info or {},
        score=0.0
    )

def store_embeddings(
    repo: QdrantEmbeddingRepository,
    video_id: str,
    keyframe_id: str,
    image_embedding: List[float] | None = None,
    text_embedding: List[float] | None = None,
    extra_info: Dict[str, Any] | None = None
) -> int:
    """Lưu các vector embedding vào kho lưu trữ."""
    if not any([image_embedding, text_embedding]):
        raise ValueError("At least one embedding (image, text) must be provided")
    
    for emb, name in [(image_embedding, "image"), (text_embedding, "text")]:
        if emb:
            expected_size = repo.vector_store.vector_sizes.get(name, 512)
            if len(emb) != expected_size:
                raise ValueError(f"{name} embedding size ({len(emb)}) does not match expected size ({expected_size})")
            if any(np.isnan(x) for x in emb):
                raise ValueError(f"{name} embedding contains invalid values (NaN)")
    
    record = create_embedding_record(
        video_id=video_id,
        keyframe_id=keyframe_id,
        image_embedding=image_embedding,
        text_embedding=text_embedding,
        extra_info=extra_info
    )
    return repo.create(record)

def search_embeddings(
    repo: QdrantEmbeddingRepository,
    image_embedding: List[float] | None = None,
    text_embedding: List[float] | None = None,
    top_k: int = 5,
    filters: Dict[str, Any] | None = None
) -> List[EmbeddingRecord]:
    """Tìm kiếm dựa trên các vector embedding theo thứ tự image -> text"""
    embeddings = {}
    if image_embedding is not None:
        embeddings["image"] = image_embedding
    if text_embedding is not None:
        embeddings["text"] = text_embedding
    
    if not embeddings:
        raise ValueError("At least one embedding (image, text) must be provided for search")
    
    return repo.search(embeddings=embeddings, top_k=top_k, filters=filters)

def delete_connectors(
    repo: QdrantEmbeddingRepository,
    record_id: int
) -> None:
    """Xóa một bản ghi embedding dựa trên ID."""
    repo.delete(record_id)

def retrive(
    repo: QdrantEmbeddingRepository,
    record_id: int
) -> EmbeddingRecord | None:
    """Lấy một bản ghi embedding dựa trên ID."""
    return repo.retrive(record_id)