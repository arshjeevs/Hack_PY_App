from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import os

# Connect to local Qdrant instance (or cloud if needed)
qdrant = QdrantClient(host="localhost", port=6333)

# Create collection if not exists
def setup_qdrant_collection():
    collection_name = "pdf_chunks"
    if not qdrant.collection_exists(collection_name):
        qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
