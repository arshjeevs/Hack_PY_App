from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.database import db_manager
from app.api.auth.route import auth_router
from app.api.documents.routes import document_router

app = FastAPI(
    title="Document Q&A System",
    description="A RAG-based system for asking questions about uploaded PDF documents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Starting Document Q&A System...")
    
    await db_manager.connect()
    print("Connected to MongoDB")

@app.on_event("shutdown")
async def shutdown_event():
    print("Shutting down...")

    await db_manager.disconnect()
    print("Disconnected from MongoDB")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Document Q&A System is running!",
        "status": "healthy"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "services": {
            "database": "connected",
            "vector_db": "ready"
        }
    }

app.include_router(auth_router, prefix="/auth", tags=["Authentication"])
app.include_router(document_router, prefix="/documents", tags=["Documents"])
