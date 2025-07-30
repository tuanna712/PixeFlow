from prep.vectordb import QdrantStore
from prep.base import Frame
import uuid
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Thông tin cloud
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "VectorDB_Test"

# 2. Khởi tạo vector store
store = QdrantStore(
    collection_name=COLLECTION_NAME,
    vector_sizes={"image": 512},
    mode="local",
    url=QDRANT_CLOUD_URL,
    api_key=QDRANT_API_KEY,
)

# 3. Tạo data mẫu
frame1 = Frame(id=str(uuid.uuid4()))
frame1.video_id = str(uuid.uuid4())
frame1.objects = ["car", "street", "London"]
frame1.caption = "A red car drives through London street"
frame1.embeddings = np.random.rand(512).tolist()

frame2 = Frame(id=str(uuid.uuid4()))
frame2.video_id = str(uuid.uuid4())
frame2.objects = ["dog", "ball", "flower"]
frame2.caption = "A dog plays with a ball in the garden, which has many flowers"
frame2.embeddings = np.random.rand(512).tolist()


# 4. Upsert vào Qdrant
store.upsert(frame1)
store.upsert(frame2)

# 5. Semantic search (tìm theo vector gần nhất)
results = store.sematic_search(
    embeddings=(np.array(frame1.embeddings) * 1.1).tolist(),
    filter={"video_id": frame1.video_id},
    top_k=3
)

print("\nSemantic search result:")
for result in results:
    print(result)

# 6. Keyword search theo object
scroll_result, _ = store.keyword_search(["car", "London"], keyword_type="objects", top_k=3)

print("\nKeyword scroll result:")
for point in scroll_result:
    print(f"ID: {point.id}, Payload: {point.payload}")

# 7. Keyword search theo caption
scroll_result, _ = store.keyword_search("red car", keyword_type="caption", top_k=3)

print("\nCaption phrase search:")
for point in scroll_result:
    print(f"ID: {point.id}, Payload: {point.payload}")

# 8. Clean-up (xóa theo video_id)
# store.remove_by_video_id(frame1.video_id)

from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

all_points = []
scroll_offset = None

while True:
    result, scroll_offset = client.scroll(
        collection_name="VectorDB_Test",
        limit=100,
        offset=scroll_offset,
        with_payload=True,
        with_vectors=False
    )
    
    all_points.extend(result)
    
    if scroll_offset is None:
        break

for i, point in enumerate(all_points):
    print(f"\n{i+1}. ID: {point.id}, Payload: {point.payload}")