import fitz  
import logging
from typing import List, Dict
from app.core.database import get_db
from app.models.document import DocumentMetadata

logger = logging.getLogger(__name__)

class DocumentService:    
    def __init__(self):
        self.chunk_size = 300
        self.overlap = 50
    
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        try:
            pdf_doc = fitz.open("pdf", pdf_bytes)
            full_text = ""
            
            for page in pdf_doc:
                full_text += page.get_text()
            
            pdf_doc.close()
            logger.info(f"Extracted {len(full_text)} characters from PDF")
            return full_text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def chunk_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.overlap
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def extract_and_chunk(self, pdf_bytes: bytes) -> List[str]:
        text = self.extract_text_from_pdf(pdf_bytes)
        return self.chunk_text(text)
    
    async def save_document_metadata(self, user_email: str, filename: str):
        try:
            db = get_db()
            doc = DocumentMetadata(user_email=user_email, filename=filename)
            await db.documents.insert_one(doc.to_dict())
            logger.info(f"Saved metadata for {filename}")
        except Exception as e:
            logger.error(f"Failed to save document metadata: {e}")
            raise
    
    async def get_user_documents(self, user_email: str) -> List[Dict]:
        try:
            db = get_db()
            documents = await db.documents.find({"user_email": user_email}).to_list(length=100)
            for doc in documents:
                doc["_id"] = str(doc["_id"])
            
            return documents
        except Exception as e:
            logger.error(f"Failed to get user documents: {e}")
            raise

document_service = DocumentService()