import logging
from typing import List
from groq import Groq
from app.core.config import settings
from app.core.vector_db import vector_db
from app.core.database import get_db
from app.models.document import Conversation
from app.services.embedding import embedding_service

logger = logging.getLogger(__name__)

class QAService:
    
    def __init__(self):
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        self.model = settings.GROQ_MODEL
        self.max_context_chunks = 5
    
    def search_relevant_chunks(self, question: str) -> List[str]:
        try:
            question_embedding = embedding_service.embed_text(question)
            
            hits = vector_db.search_similar(
                query_vector=question_embedding,
                limit=self.max_context_chunks
            )
            
            relevant_chunks = [hit.payload["chunk_text"] for hit in hits]
            logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Failed to search for relevant chunks: {e}")
            raise
    
    def generate_answer(self, question: str, context_chunks: List[str]) -> str:
        try:
            context = "\n---\n".join(context_chunks)
            
            prompt = f"""
You are a helpful assistant. Use the context below to answer the question.
If the context doesn't contain enough information, say so.

Context:
{context}

Question: {question}

Answer:
"""
            
            response = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            
            answer = response.choices[0].message.content.strip()
            logger.info("Generated answer successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise
    
    async def save_conversation(self, user_email: str, question: str, answer: str, filename: str = None):
        try:
            db = get_db()
            conversation = Conversation(
                user_email=user_email,
                filename=filename,
                question=question,
                answer=answer
            )
            await db.conversations.insert_one(conversation.to_dict())
            logger.info("Saved conversation to database")
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")
            raise
    
    async def answer_question(self, question: str, user_email: str, filename: str = None) -> str:
        try:
            relevant_chunks = self.search_relevant_chunks(question)
            
            answer = self.generate_answer(question, relevant_chunks)
            
            await self.save_conversation(user_email, question, answer, filename)
            
            return answer
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            raise
    
    async def get_user_conversations(self, user_email: str) -> List[dict]:
        try:
            db = get_db()
            conversations = await db.conversations.find({"user_email": user_email}).to_list(length=100)
            
            for conv in conversations:
                conv["_id"] = str(conv["_id"])
            
            return conversations
        except Exception as e:
            logger.error(f"Failed to get user conversations: {e}")
            raise

qa_service = QAService()
