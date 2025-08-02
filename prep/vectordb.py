from qdrant_client import QdrantClient, models
from typing import Dict, Optional
from loguru import logger
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]

from prep.base import Frame

class QdrantStore:
    def __init__(self,
                 collection_name: str,
                 vector_sizes: Dict[str, int],
                 mode: str = "local",
                 host: Optional[str] = "localhost",
                 port: Optional[int] = 6333,
                 url: Optional[str] = None,
                 api_key: Optional[str] = None):
        
        self.collection_name = collection_name
        self.vector_sizes = vector_sizes
        
        if mode == "local":
            self.client = QdrantClient(host=host, port=port)
        elif mode == "cloud":
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            logger.error("Invalid mode. Use 'local' or 'cloud'.")
            raise ValueError("Invalid mode. Use 'local' or 'cloud'.")
        
        self._create_collection()
    
    def _create_collection(self):
        is_exists = self.client.collection_exists(collection_name=self.collection_name)
        if is_exists:
            logger.info(f"Collection {self.collection_name} already exists. Skip creation.")
            return
        
        logger.info(f"Collection {self.collection_name} not found. Creating new collection.")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                                size=self.vector_sizes.get("image", 512),
                                distance=models.Distance.COSINE
                            ),
        )
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="video_id",
            field_schema="uuid",
        )
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="objects",
            field_schema="keyword",
        )
        
        self.client.create_payload_index(
            collection_name=self.collection_name,
            field_name="caption",
            field_schema=models.TextIndexParams(
                type="text",
                tokenizer=models.TokenizerType.WORD,
                lowercase=True,
                phrase_matching=True,
            ),
        )
    
    def upsert(self, frame: Frame):
        frame_id = frame.id
        embeddings = frame.embeddings
        payload = {
            "video_id": frame.video_id,
            "objects": frame.objects,
            "caption": frame.caption
        }
        
        point = [
            models.PointStruct(
                id=frame_id,
                payload=payload,
                vector=embeddings
            )
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=point
        )
        
        logger.info(f"Upserted frame {frame_id} into collection {self.collection_name}.")
            
    def delete(self, point_ids):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=point_ids,
            ),
        )
    
    def retrieve(self, point_ids):
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_ids],
        )
        return points if points else None
    
    def remove_by_video_id(self, video_id):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="video_id",
                            match=models.MatchValue(value=video_id),
                        ),
                    ],
                )
            )
        )
    
    def sematic_search(self, embeddings, filter=None, top_k=10, with_payload=True):
        
        if filter:
            must = [
                models.FieldCondition(
                    key=k,
                    match=models.MatchValue(value=v)
                ) for k, v in filter.items()
            ]
            query_filter = models.Filter(must=must)
        else:
            query_filter = None
        
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=embeddings,
            query_filter=query_filter,
            search_params=models.SearchParams(hnsw_ef=128, exact=False),
            limit=top_k,
            with_payload=with_payload,
        )
        
        return [{"frame_id": point.id, "payload": point.payload} for point in results.points]
    
    def keyword_search(self, keywords, keyword_type="objects", top_k=10):
        
        if keyword_type == "objects":
            if not isinstance(keywords, list):
                logger.error("Keywords for 'objects' must be a list.")
                raise ValueError("Keywords for 'objects' must be a list.")
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="objects",
                            match=models.MatchAny(any=keywords),
                        )
                    ]
                ),
                limit=top_k
            )
            
            return results
        
        elif keyword_type == "caption":
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="caption",
                            match=models.MatchPhrase(phrase=str(keywords)),
                        )
                    ]
                ),
                limit=top_k
            )
            
            return results
        
        else:
            logger.error("Invalid keyword type. Use 'objects' or 'caption'.")