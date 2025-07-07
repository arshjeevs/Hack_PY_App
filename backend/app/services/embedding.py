from sentence_transformers import SentenceTransformer
from uuid import uuid4
from app.core.vector_db import qdrant

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_and_store_chunks(chunks: list[str], metadata: dict):
    vectors = model.encode(chunks).tolist()

    # Each chunk = (id, vector, payload)
    qdrant_points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        qdrant_points.append({
            "id": str(uuid4()),
            "vector": vector,
            "payload": {
                "chunk_text": chunk,
                **metadata
            }
        })

    qdrant.upsert(
        collection_name="pdf_chunks",
        points=qdrant_points
    )

    return len(qdrant_points)
