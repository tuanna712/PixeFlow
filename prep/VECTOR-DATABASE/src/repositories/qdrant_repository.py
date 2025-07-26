from typing import Any, Dict, List
from pathlib import Path
from loguru import logger
import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from repositories.repository import EmbeddingRepository
from models import EmbeddingRecord
from adapter.sqlite_store import SQLiteMetadataStore
from adapter.qdrant_store import QdrantVectorStore

class QdrantEmbeddingRepository(EmbeddingRepository):
    def __init__(self, vector_store: QdrantVectorStore, metadata_store: SQLiteMetadataStore):
        self.vector_store = vector_store
        self.metadata_store = metadata_store
    
    def create(self, record: EmbeddingRecord) -> int:
        new_id = self.metadata_store.insert(record)
        self.vector_store.upsert(
            record_id=new_id,
            embeddings=record.embeddings,
            payload={
                "video_id": record.video_id,
                "keyframe_id": record.keyframe_id,
                "labels": record.extra_info.get("labels", []),
            }
        )
        return new_id
    
    def update(self, record: EmbeddingRecord) -> None:
        self.metadata_store.update(record)
        self.vector_store.upsert(
            record_id=record.id,
            embeddings=record.embeddings,
            payload={
                "video_id": record.video_id,
                "keyframe_id": record.keyframe_id,
                "labels": record.extra_info.get("labels", []),
            }
        )
    
    def delete(self, record_id: int) -> None:
        self.metadata_store.delete(record_id)
        self.vector_store.delete(record_id)
        logger.info(f"Deleted record with ID: {record_id}")
    
    def retrive(self, record_id: int) -> EmbeddingRecord | None:
        record = self.metadata_store.get(record_id)
        if record:
            record.embeddings = self.vector_store.retrieve(record_id)
            return record
        logger.warning(f"Record with ID {record_id} not found")
        return None
    
    def search(self, embeddings: Dict[str, List[float]], top_k: int,
               filters: Dict[str, Any] | None = None) -> List[EmbeddingRecord]:
        hits = self.vector_store.search(embeddings, top_k, filters)
        results = []
        for db_id, score in hits:
            meta_data = self.metadata_store.get(db_id)
            if meta_data:
                meta_data.embeddings = {}
                meta_data.score = score
                results.append(meta_data)
        return results