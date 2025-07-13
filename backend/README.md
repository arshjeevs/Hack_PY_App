# Document Q&A System - Backend

A **RAG (Retrieval Augmented Generation)** system that allows users to upload PDF documents and ask questions about them using AI.

## 🏗️ System Architecture

### Overview
This is a **FastAPI-based backend** that implements a complete RAG pipeline:

1. **Document Upload** → PDF text extraction and chunking
2. **Vector Storage** → Text embeddings stored in Qdrant vector database
3. **Question Answering** → Semantic search + LLM generation
4. **Data Persistence** → MongoDB for metadata and conversation history

### Key Components

#### 📁 Core Structure
```
backend/
├── app/
│   ├── core/           # Configuration and database connections
│   ├── models/         # Data models (Pydantic)
│   ├── services/       # Business logic
│   ├── api/           # FastAPI routes
│   └── schemas/       # Request/response schemas
├── main.py            # Application entry point
└── requirements.txt   # Dependencies
```

#### 🔧 Core Services

**1. Database Manager (`core/database.py`)**
- Manages MongoDB connections
- Handles connection errors and logging
- Provides clean interface for database operations

**2. Vector Database Manager (`core/vector_db.py`)**
- Manages Qdrant vector database
- Handles vector storage and similarity search
- Creates collections automatically

**3. Configuration (`core/config.py`)**
- Centralized settings management
- Environment variable handling
- Clean separation of concerns

#### 🚀 Business Logic Services

**1. Document Service (`services/document.py`)**
- PDF text extraction using PyMuPDF
- Text chunking with overlap for better context
- Document metadata management

**2. Embedding Service (`services/embedding.py`)**
- Text-to-vector conversion using SentenceTransformers
- Batch embedding processing
- Vector storage coordination

**3. QA Service (`services/qa.py`)**
- Complete RAG pipeline implementation
- Semantic search for relevant chunks
- LLM integration with Groq
- Conversation history management

#### 🌐 API Layer

**Document Routes (`api/documents/routes.py`)**
- `POST /upload` - Upload and process PDF
- `POST /ask` - Ask questions about documents
- `GET /files` - Get user's documents
- `GET /conversations` - Get conversation history

## 🔄 RAG Pipeline Flow

### 1. Document Processing
```
PDF Upload → Text Extraction → Chunking → Embedding → Vector Storage
```

### 2. Question Answering
```
Question → Embedding → Vector Search → Context Retrieval → LLM Generation → Answer
```

## 🛠️ Technologies Used

- **FastAPI** - Modern Python web framework
- **MongoDB** - Document database for metadata
- **Qdrant** - Vector database for embeddings
- **SentenceTransformers** - Text embedding model
- **Groq** - Fast LLM API for text generation
- **PyMuPDF** - PDF text extraction

## 🚀 How to Explain in Interview

### 1. System Overview (30 seconds)
"This is a RAG-based document Q&A system. Users upload PDFs, the system extracts text, creates embeddings, and stores them in a vector database. When users ask questions, it finds relevant text chunks and uses an LLM to generate answers."

### 2. Architecture (1 minute)
"The system follows a clean layered architecture:
- **API Layer**: FastAPI routes handle HTTP requests
- **Service Layer**: Business logic for document processing, embeddings, and QA
- **Data Layer**: MongoDB for metadata, Qdrant for vectors
- **Core Layer**: Database connections and configuration"

### 3. RAG Implementation (1 minute)
"RAG works in two phases:
- **Indexing**: PDF → text chunks → embeddings → vector storage
- **Querying**: Question → embedding → vector search → context → LLM → answer"

### 4. Key Design Decisions (30 seconds)
"- **Service-oriented architecture** for maintainability
- **Error handling and logging** throughout
- **Async/await** for better performance
- **Clean separation** of concerns"

### 5. Scalability Considerations (30 seconds)
"- Vector database can handle millions of embeddings
- MongoDB for horizontal scaling
- Stateless services for easy deployment
- Environment-based configuration"

## 🔧 Quick Setup

### Option 1: Automated Setup (Recommended)
```bash
# Run the setup script
python setup.py

# Start MongoDB (if not running)
mongod

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Run the application
uvicorn main:app --reload
```

### Option 2: Manual Setup
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create .env file**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start services**:
   ```bash
   # MongoDB
   mongod
   
   # Qdrant
   docker run -p 6333:6333 qdrant/qdrant
   ```

4. **Run the application**:
   ```bash
   uvicorn main:app --reload
   ```

## 📊 API Endpoints

- `GET /` - Health check
- `GET /health` - Detailed health status
- `POST /documents/upload` - Upload PDF
- `POST /documents/ask` - Ask question
- `GET /documents/files` - Get user documents
- `GET /documents/conversations` - Get conversation history

## 🎯 Key Features

- ✅ PDF text extraction and chunking
- ✅ Vector-based similarity search
- ✅ RAG-powered question answering
- ✅ Conversation history tracking
- ✅ Error handling and logging
- ✅ Clean, maintainable code structure
- ✅ Interview-ready documentation 