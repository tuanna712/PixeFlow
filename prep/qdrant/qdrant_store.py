from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Any, Dict, List, Tuple
import numpy as np
from loguru import logger

class QdrantVectorStore:
    def __init__(self, url: str, collection_name: str, 
                 vector_sizes: Dict[str, int]):
        self.client = QdrantClient(url=url, prefer_grpc=False)
        self.collection = collection_name
        self.vector_sizes = vector_sizes
        if collection_name == "FrameClipEmbedding":
            self.dim = 512
        elif collection_name == "FrameBeitEmbedding":
            self.dim = 1024
        self._create_collection()
    
    def _create_collection(self):
        is_exists = self.client.collection_exists(collection_name=self.collection)
        if is_exists:
            logger.info(f"Collection '{self.collection}' already exists. Skip creation.")
            return
        
        logger.info(f"Collection '{self.collection}' not found. Creating new collection.")
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "image": models.VectorParams(
                    size=self.vector_sizes.get("image", self.dim),
                    distance=models.Distance.COSINE
                    # Leave HNSW indexing ON for image
                ),
                "text": models.VectorParams(
                    size=self.vector_sizes.get("text", self.dim),
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0) # Disable HNSW for reranking
                )   
            }
        )
        
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="video_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="frame_id",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
    
    def upsert(self, record_id: int, embeddings: Dict[str, List[float]], payload: Dict[str, Any]):
        for vec_type, size in self.vector_sizes.items():
            if vec_type in embeddings and len(embeddings[vec_type]) != size:
                raise ValueError(f"{vec_type} embedding size ({len(embeddings[vec_type])}) does not match expected size ({size})")
            if vec_type in embeddings and any(np.isnan(x) for x in embeddings[vec_type]):
                raise ValueError(f"{vec_type} embedding contains invalid values (NaN)")
        
        points=[
            models.PointStruct(
                id=record_id,
                vector={
                    "image": embeddings.get("image", []),
                    "text": embeddings.get("text", [])
                },
                payload=payload
            )
        ]
        self.client.upload_points(
            collection_name=self.collection,
            points=points
        )

    def _validate_vector(self, vec_type: str, vector: List[float]):
        expected_size = self.vector_sizes.get(vec_type)
        if expected_size is None:
            raise ValueError(f"Unknown vector type: {vec_type}")
        if len(vector) != expected_size:
            raise ValueError(f"{vec_type} embedding size ({len(vector)}) does not match expected size ({expected_size})")
        if any(np.isnan(x) for x in vector):
            raise ValueError(f"{vec_type} embedding contains invalid values (NaN)")

    def upsert_text_vector(self, record_id: int, text_vector: List[float], payload: Dict[str, Any] = None):
        self._validate_vector("text", text_vector)
        points = [
            models.PointStruct(
                id=record_id,
                vector={"text": text_vector},
                payload=payload
            )
        ]
        self.client.upload_points(collection_name=self.collection, points=points)

    def upsert_image_vector(self, record_id: int, image_vector: List[float], payload: Dict[str, Any] = None):
        self._validate_vector("image", image_vector)
        points = [
            models.PointStruct(
                id=record_id,
                vector={"image": image_vector},
                payload=payload
            )
        ]
        self.client.upload_points(collection_name=self.collection, points=points)
    
    def delete(self, record_id: int):
        delete_vectors = ["image", "text"]
        self.client.delete_vectors(
            collection_name=self.collection,
            points=[record_id],
            vectors=delete_vectors,
        )

    def count(self) -> int:
        return self.client.count(collection_name=self.collection,
                                exact=True,
                            ).count

    def retrieve(self, record_id: int) -> models.PointStruct | None:
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=[record_id],
        )
        return points[0].vector if points else None
    
    def search(self, embeddings: Dict[str, List[float]], top_k: int, 
               filters: Dict[str, Any] | None = None) -> List[Tuple[int, float]]:
        if not embeddings:
            raise ValueError("At least one embedding must be provided for search")
        
        for vec_type, emb in embeddings.items():
            if vec_type not in self.vector_sizes:
                raise ValueError(f"Invalid vector type: {vec_type}")
            if len(emb) != self.vector_sizes[vec_type]:
                raise ValueError(f"{vec_type} embedding size ({len(emb)}) does not match expected size ({self.vector_sizes[vec_type]})")
            if any(np.isnan(x) for x in emb):
                raise ValueError(f"{vec_type} embedding contains invalid values (NaN)")
        
        if filters:
            must = [
                models.FieldCondition(
                    key=k,
                    match=models.MatchValue(value=v)
                ) for k, v in filters.items()
            ]
            query_filter = models.Filter(must=must)
        else:
            query_filter = None
        
        # Thứ tự tìm kiếm tuần tự: image -> text
        search_order = ["image", "text"]
        available_vectors = [vt for vt in search_order if vt in embeddings]
        
        if not available_vectors:
            raise ValueError("No valid vectors provided for search")
        
        if len(available_vectors) == 1 and available_vectors[0] == "image":
            current_hits = self.search_only_image(embeddings, top_k, query_filter, available_vectors)
        elif len(available_vectors) == 1 and available_vectors[0] == "text":
            current_hits = self.search_only_text(embeddings, top_k, query_filter, available_vectors)
        elif len(available_vectors) == 2:
            current_hits = self.search_image_text(embeddings, top_k, query_filter, available_vectors)
        else:
            raise ValueError("Only image or (image, text) vectors are supported for search")
        if not current_hits:
            return []
        
        return current_hits[:top_k]
    
    def search_only_image(self, embeddings: Dict[str, List[float]], top_k: int, 
               query_filter: models.Filter | None = None, available_vectors: List[str] = ["image"]) -> List[Tuple[int, float]]:
        current_hits = None
        current_hits = self.client.query_points(
            collection_name=self.collection,
            query=embeddings[available_vectors[0]],
            using=available_vectors[0],
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        ).points
        current_hits = [(hit.id, hit.score) for hit in current_hits]
        return current_hits
    
    def search_only_text(self, embeddings: Dict[str, List[float]], top_k: int, 
               query_filter: models.Filter | None = None, available_vectors: List[str] = ["text"]) -> List[Tuple[int, float]]:
        current_hits = None
        current_hits = self.client.query_points(
            collection_name=self.collection,
            query=embeddings[available_vectors[0]],
            using=available_vectors[0],
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        ).points
        current_hits = [(hit.id, hit.score) for hit in current_hits]
        return current_hits
    
    def search_image_text(self, embeddings: Dict[str, List[float]], top_k: int, 
               query_filter: models.Filter | None = None, available_vectors: List[str] = ["image", "text"]) -> List[Tuple[int, float]]:
        current_hits = self.client.query_points(
            collection_name=self.collection,
            prefetch=models.Prefetch(
                query=embeddings[available_vectors[0]],
                using=available_vectors[0],
            ),
            query=embeddings[available_vectors[1]],
            using=available_vectors[1],
            limit=top_k,
            query_filter=query_filter,
            with_payload=True
        ).points
        current_hits = [(hit.id, hit.score) for hit in current_hits]
        return current_hits
