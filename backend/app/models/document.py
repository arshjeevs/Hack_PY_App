from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class DocumentMetadata(BaseModel):
    user_email: EmailStr
    filename: str
    uploaded_at: datetime = datetime.utcnow()
    
    def to_dict(self):
        return {
            "user_email": self.user_email,
            "filename": self.filename,
            "uploaded_at": self.uploaded_at
        }

class Conversation(BaseModel):
    user_email: EmailStr
    filename: Optional[str] = None
    question: str
    answer: str
    timestamp: datetime = datetime.utcnow()
    
    def to_dict(self):
        return {
            "user_email": self.user_email,
            "filename": self.filename,
            "question": self.question,
            "answer": self.answer,
            "timestamp": self.timestamp
        }
