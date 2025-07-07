from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document import extract_text_and_chunk
from app.services.qa import answer_question
from app.services.embedding import embed_and_store_chunks
from app.core.vector_db import setup_qdrant_collection
from app.schemas.document_schema import QueryRequest

document_router = APIRouter()

@document_router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    chunks = extract_text_and_chunk(content)

    setup_qdrant_collection()

    metadata = {"filename": file.filename}
    added = embed_and_store_chunks(chunks, metadata)

    return {
        "filename": file.filename,
        "chunks_stored": added
    }

@document_router.post("/ask")
async def ask_question(data: QueryRequest):
    try:
        response = answer_question(data.question)
        return {"question": data.question, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
