import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.mongo_uri = "mongodb+srv://admin:admin123@cluster0.nim4yef.mongodb.net/"
        self.db_name = os.getenv("MONGODB_NAME", "document_qa")
    
    async def connect(self):
        try:
            logger.info(f"MONGO_URI : {self.mongo_uri}")
            self.client = AsyncIOMotorClient(self.mongo_uri)
            await self.client.server_info()
            self.db = self.client[self.db_name]
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise RuntimeError(f"Database connection failed: {e}")
    
    async def disconnect(self):
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    def get_db(self):
        if self.db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self.db

db_manager = DatabaseManager()

def get_db():
    return db_manager.get_db() 