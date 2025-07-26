from models import EmbeddingRecord
from adapter.sqlite_store import SQLiteMetadataStore
from adapter.qdrant_store import QdrantVectorStore
from repositories.qdrant_repository import QdrantEmbeddingRepository
from embedding_connectors import store_embeddings, search_embeddings, delete_connectors
import yaml
import numpy as np
from loguru import logger
from typing import List, Dict
from pathlib import Path
import os
from dotenv import load_dotenv

root_path = Path(__file__).resolve().parents[1]
logger.add(root_path / "logs/main.log", rotation="7 days", compression="zip", level="INFO")

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_repo(config_path: str) -> QdrantEmbeddingRepository:
    
    load_dotenv()
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    config = load_config(config_path)
    
    sqlite_store = SQLiteMetadataStore(config.get("DB_PATH", "metadata.db"))
    collection_name = config.get("COLLECTION_NAME")
    vector_sizes = config.get("VECTOR_SIZES", {"image": 512, "text": 512})
    
    vector_store = QdrantVectorStore(
        url=url,
        api_key=api_key,
        collection_name=collection_name,
        vector_sizes=vector_sizes
    )
    
    repo = QdrantEmbeddingRepository(
        vector_store=vector_store,
        metadata_store=sqlite_store
    )
    return repo

def add(repo: QdrantEmbeddingRepository, num_samples: int = 1) -> List[List[float]]:
    
    np.random.seed(42)
    query_list =[]
    
    image_dim = repo.vector_store.vector_sizes["image"]
    text_dim = repo.vector_store.vector_sizes["text"]

    for i in range(601, 601+num_samples):
        image_embedding = np.random.normal(0, 1, size=image_dim).tolist()
        text_embedding = np.random.normal(0, 1, size=text_dim).tolist()

        new_id = store_embeddings(
            repo=repo,
            video_id=f"vid_{i+1}",
            keyframe_id=f"img_{i+1}",
            image_embedding=image_embedding,
            text_embedding=text_embedding,
            extra_info={"caption": f"Sample #{i+1}"}
        )
        logger.info(f"Inserted record {i+1}/{num_samples} with ID: {new_id}")
        
        if i % 50 == 0:
            query_list.append([image_embedding, text_embedding])
    
    return query_list

def query(repo: QdrantEmbeddingRepository, vector_query: Dict[str, List[float]], top_k: int = 3, filter: Dict = None) -> None:
    image_embedding = vector_query["image"] if len(vector_query) > 0 else None
    text_embedding = vector_query["text"] if len(vector_query) > 1 else None
    
    # Truy vấn chỉ với embedding ảnh
    search_results = search_embeddings(
        repo=repo,
        image_embedding=image_embedding,
        top_k=top_k,
    )
    logger.info("Search results (image only):")
    for res in search_results:
        logger.info(f"Record: {res}, Score: {res.score}")
    
    # Truy vấn chỉ với embedding text
    search_results = search_embeddings(
        repo=repo,
        text_embedding=text_embedding,
        top_k=3,
    )
    logger.info("Search results (text only):")
    for res in search_results:
        logger.info(f"Record: {res}, Score: {res.score}")
    
    # Truy vấn với tất cả embedding
    search_results = search_embeddings(
        repo=repo,
        image_embedding=image_embedding,
        text_embedding=text_embedding,
        top_k=3,
    )
    logger.info("Search results (all embeddings):")
    for res in search_results:
        logger.info(f"Record: {res}, Score: {res.score}")

def retrive(repo: QdrantEmbeddingRepository, record_id: int) -> EmbeddingRecord | None:
    record = repo.retrive(record_id)
    logger.info(f"Retrieved record: {record}")

def main():
    config_path = "../config.yml"
    repo = get_repo(config_path)
    query_list = add(repo, num_samples=300)
    for i, q in enumerate(query_list):
        vector_query = {
            "image": (np.array(q[0]) * 1.03).tolist(),
            "text": (np.array(q[1]) * 1.1).tolist()
        }
        top_k = 3
        
        logger.info(f"=========================Query {i+1}th vector=========================")
        query(repo, vector_query, top_k)
        logger.info("\n")
    
    retrive(repo, record_id=620)

if __name__ == "__main__":
    main()
    # from qdrant_client import QdrantClient

    # client = QdrantClient(
    #     url="https://d879cf11-4168-4623-8c79-8ae43b5ca7b6.europe-west3-0.gcp.cloud.qdrant.io:6333",
    #     api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.nEFLnpmEPzcyXvOIl07YCZz31m44Y4YRg7QC_0rAvYU"
    # )

    # points =client.retrieve(
    #     collection_name="VectorDB_AI_Challenge_2025",
    #     ids=[5, 105, 205],
    # )
    # print(points)
