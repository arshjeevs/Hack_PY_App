"""
Embedding Service
Handles text embedding and vector storage operations
"""
import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.core.vector_db import vector_db

logger = logging.getLogger(__name__)

class EmbeddingService:
    
    def __init__(self):
        # Load the embedding model once
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
    
    def embed_text(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise
    
    def embed_chunks(self, chunks: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(chunks)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to embed chunks: {e}")
            raise
    
    def store_chunks_with_embeddings(self, chunks: List[str], metadata: Dict) -> int:
        try:
            vector_db.setup_collection()
            
            embeddings = self.embed_chunks(chunks)
            
            next_id = vector_db.get_next_id()
            
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point = {
                    "id": next_id + i, 
                    "vector": embedding,
                    "payload": {
                        "chunk_text": chunk,
                        "filename": metadata["filename"],
                        "chunk_index": i
                    }
                }
                points.append(point)
            
            vector_db.add_vectors(points)
            logger.info(f"Stored {len(points)} chunks in vector database")
            return len(points)
            
        except Exception as e:
            logger.error(f"Failed to store chunks: {e}")
            raise

embedding_service = EmbeddingService()
