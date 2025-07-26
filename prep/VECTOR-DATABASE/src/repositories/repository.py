from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from models import EmbeddingRecord

class EmbeddingRepository(ABC):
    @abstractmethod
    def create(self, record: EmbeddingRecord) -> int:
        """Tạo một bản ghi embedding mới."""
        pass
    
    @abstractmethod
    def update(self, record: EmbeddingRecord) -> None:
        """Cập nhật một bản ghi embedding."""
        pass

    @abstractmethod
    def delete(self, record_id: int) -> None:
        """Xóa một bản ghi embedding."""
        pass

    @abstractmethod
    def search(self, embeddings: Dict[str, List[float]], top_k: int,
               filters: Dict[str, Any] | None = None,
               weights: Dict[str, float] | None = None,
               prefetch_vector_type: str | None = None) -> List[EmbeddingRecord]:
        """Tìm kiếm các bản ghi embedding."""
        pass