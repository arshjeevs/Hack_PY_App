import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorDBManager:
    
    def __init__(self):
        self.client = QdrantClient(
            host=settings.QDRANT_HOST, 
            port=settings.QDRANT_PORT
        )
        self.collection_name = "pdf_chunks"
        self.vector_size = 384
    
    def setup_collection(self):
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, 
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Failed to setup collection: {e}")
            raise
    
    def search_similar(self, query_vector, limit=5):
        try:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            return hits
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def get_next_id(self):
        try:
            points = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Get all points
                with_payload=False,
                with_vectors=False
            )[0]
            
            if not points:
                return 0
            
            ids = []
            for point in points:
                try:
                    if isinstance(point.id, int):
                        ids.append(point.id)
                    elif isinstance(point.id, str) and point.id.isdigit():
                        ids.append(int(point.id))
                except (ValueError, AttributeError):
                    continue
            
            if not ids:
                return 0
            
            max_id = max(ids)
            return max_id + 1
        except Exception as e:
            logger.error(f"Failed to get next ID: {e}")
            return 0
    
    def add_vectors(self, points):
        """Add vectors to the collection"""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
            logger.info(f"Added {len(points)} vectors to collection")
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise

vector_db = VectorDBManager()
