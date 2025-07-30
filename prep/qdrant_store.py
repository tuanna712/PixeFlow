from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Any, Dict, List, Tuple
import numpy as np
from loguru import logger

class QdrantVectorStore:
    def __init__(self, 
                 url: str=None,
                 path: str=None, 
                 api_key=None, 
                 collection_name: str = None, 
                 mode="onprem",
                 ):
        if mode == "onprem":
            if not path:
                raise ValueError("Path must be provided for on-premise mode")
            self.client = QdrantClient(path=path)
        elif mode == "cloud":
            if not api_key:
                raise ValueError("API key must be provided for cloud mode")
            if not url:
                raise ValueError("URL must be provided for cloud mode")
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            raise ValueError("Invalid mode. Use 'onprem' or 'cloud'.")
        self.collection = collection_name
        if not self.collection:
            raise ValueError("Collection name must be provided")
        collections = self.client.get_collections().collections
        if self.collection not in collections:
            logger.warning(f"Collection '{self.collection}' does not exist. It will be created on first upsert.")

    def create_collection(self, vector_sizes: Dict[str, int] = {"image": 512, "text": 512}):
        is_exists = self.client.collection_exists(collection_name=self.collection)
        if is_exists:
            logger.info(f"Collection '{self.collection}' already exists. Skip creation.")
            return

        logger.info(f"Collection '{self.collection}' not found. Creating new collection.")
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                "image": models.VectorParams(
                    size=vector_sizes.get("image", 512),
                    distance=models.Distance.COSINE
                    # Leave HNSW indexing ON for image
                ),
                "text": models.VectorParams(
                    size=vector_sizes.get("text", 512),
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0) # Disable HNSW for reranking
                )   
            }
        )
        
    def set_payload(self):
        if not self.client.collection_exists(collection_name=self.collection):
            raise ValueError(f"Collection '{self.collection}' does not exist. Create it before adding payload indices.")
        self.client.set_payload(
            collection_name=self.collection,
            payload={
                "video_id": "string",
                "frame_index": "int",
                "ocr": "string",
                "objects": "list",
                "transcription": "string",
                "description": "string",
                "caption": "string",
            },
        )

    def delete_collection(self):
        if not self.client.collection_exists(collection_name=self.collection):
            logger.info(f"Collection '{self.collection}' does not exist. Skip dropping.")
            return

        logger.info(f"Dropping collection '{self.collection}'.")
        self.client.delete_collection(collection_name=self.collection)
    
    def upsert(self, record_id: int, embeddings: Dict[str, List[float]], 
               payload: Dict[str, Any], vector_sizes: Dict[str, int] = {"image": 512, "text": 512}):
        for vec_type, size in vector_sizes.items():
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
    
    def delete(self, record_id: int):
        delete_vectors = ["image", "text"]
        self.client.delete_vectors(
            collection_name=self.collection,
            points=[record_id],
            vectors=delete_vectors,
        )
    
    def retrieve(self, record_id: int) -> models.PointStruct | None:
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=[record_id],
        )
        return points[0].vector if points else None
    
    def search(self, 
               embeddings: Dict[str, List[float]], 
               top_k: int, 
               filters: Dict[str, Any] | None = None,
               vector_sizes: Dict[str, int] = {"image": 512, "text": 512}
               ) -> List[Tuple[int, float]]:
        if not embeddings:
            raise ValueError("At least one embedding must be provided for search")
        
        for vec_type, emb in embeddings.items():
            if vec_type not in vector_sizes:
                raise ValueError(f"Invalid vector type: {vec_type}")
            if len(emb) != vector_sizes[vec_type]:
                raise ValueError(f"{vec_type} embedding size ({len(emb)}) does not match expected size ({vector_sizes[vec_type]})")
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

    def get_collection_info(self, collection_name: str) -> models.CollectionInfo:
        if not self.client.collection_exists(collection_name=collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")  
        
        return self.client.get_collection_info(collection_name=collection_name)
    
    def get_collections(self) -> List[str]:
        return self.client.get_collections().collections

    def close(self):
        self.client.close()