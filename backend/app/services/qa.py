from app.core.vector_db import qdrant
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# Load model + Groq client
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
groq_model = os.getenv("GROQ_MODEL", "llama3-8b-8192")

def answer_question(question: str):
    # Step 1: Embed question
    question_embedding = model.encode(question).tolist()

    # Step 2: Search in Qdrant
    hits = qdrant.search(
        collection_name="pdf_chunks",
        query_vector=question_embedding,
        limit=5
    )

    # Step 3: Combine context from top chunks
    context = "\n---\n".join(hit.payload["chunk_text"] for hit in hits)

    # Step 4: Create RAG prompt
    prompt = f"""
You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {question}

Answer:
"""

    # Step 5: Send to Groq LLM
    response = groq.chat.completions.create(
        model=groq_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()
