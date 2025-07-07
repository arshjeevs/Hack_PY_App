import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

class Settings:
    JWT_SECRET = os.getenv("JWT_SECRET_KEY")
    MONGO_URI = os.getenv("MONGODB_URI")
    DB_NAME = os.getenv("MONGODB_NAME")

settings = Settings()

# Global DB variable
db = None

async def connect_to_db():
    global db
    client = AsyncIOMotorClient(settings.MONGO_URI)
    db = client[settings.DB_NAME]
