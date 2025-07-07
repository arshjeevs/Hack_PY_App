from fastapi import FastAPI
from app.api.auth.route import auth_router
from app.core.config import connect_to_db
from fastapi.middleware.cors import CORSMiddleware
from app.api.documents.routes import document_router

app = FastAPI()

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to MongoDB at startup
@app.on_event("startup")
async def startup():
    await connect_to_db()

@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI is running!"}

# Add this line below auth_router
app.include_router(document_router, prefix="/documents", tags=["Documents"])

# Include authentication routes
app.include_router(auth_router, prefix="/auth", tags=["Auth"])
