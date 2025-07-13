from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from app.services.document import document_service
from app.services.qa import qa_service
from app.services.embedding import embedding_service
from app.schemas.document_schema import QueryRequest

document_router = APIRouter()

@document_router.get("/files")
async def get_user_files(email: str = Query(..., description="User email")):
    try:
        documents = await document_service.get_user_documents(email)
        return {"documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get documents: {str(e)}")

@document_router.get("/conversations")
async def get_user_conversations(email: str = Query(..., description="User email")):
    try:
        conversations = await qa_service.get_user_conversations(email)
        return {"conversations": conversations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversations: {str(e)}")

@document_router.post("/upload")
async def upload_pdf(file: UploadFile = File(..., description="PDF file to upload")):
    try:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        content = await file.read()
        
        chunks = document_service.extract_and_chunk(content)
        
        metadata = {"filename": file.filename}
        chunks_stored = embedding_service.store_chunks_with_embeddings(chunks, metadata)
        
        await document_service.save_document_metadata(
            user_email="test@example.com",
            filename=file.filename
        )
        
        return {
            "message": "Document uploaded successfully",
            "filename": file.filename,
            "chunks_stored": chunks_stored
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@document_router.post("/ask")
async def ask_question(data: QueryRequest):
    try:
        answer = await qa_service.answer_question(
            question=data.question,
            user_email="test@example.com",  # TODO: Get from authentication
            filename=data.filename if hasattr(data, 'filename') else None
        )
        
        return {
            "question": data.question,
            "answer": answer
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")
