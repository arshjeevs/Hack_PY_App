import streamlit as st
import fitz  # PyMuPDF
import numpy as np
from typing import List, Tuple
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import re
from dataclasses import dataclass
import pickle
import os
from pathlib import Path

# Configuration
CACHE_DIR = Path("cache")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_DIMENSION = 384  # Dimension for MiniLM embeddings
TOP_K_MATCHES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models
@st.cache_resource
def load_models():
    # Load embedding model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=DEVICE)
    
    # Load LLM and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    
    return embedding_model, model, tokenizer

@dataclass
class Document:
    text: str
    embedding: np.ndarray = None

def create_cache_dir():
    """Create cache directory if it doesn't exist"""
    CACHE_DIR.mkdir(exist_ok=True)

def get_cache_path(filename: str) -> Path:
    """Get path for cached embeddings"""
    return CACHE_DIR / f"{filename}.pkl"

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from uploaded PDF file"""
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None

def preprocess_text(text: str) -> str:
    """Clean and preprocess extracted text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end < text_length:
            while end > start and text[end] != ' ':
                end -= 1
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end - overlap

    return chunks

def get_embeddings(texts: List[str], embedding_model: SentenceTransformer) -> List[np.ndarray]:
    """Generate embeddings using sentence-transformers"""
    try:
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        return [np.array(embedding) for embedding in embeddings]
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

def create_faiss_index(embeddings: List[np.ndarray]) -> faiss.IndexFlatL2:
    """Create and populate FAISS index"""
    embeddings_array = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(EMBED_DIMENSION)
    index.add(embeddings_array)
    return index

def get_relevant_chunks(query: str, chunks: List[str], faiss_index: faiss.IndexFlatL2, embedding_model: SentenceTransformer) -> List[str]:
    """Retrieve most relevant chunks for the query"""
    query_embedding = embedding_model.encode([query])[0]
    D, I = faiss_index.search(
        np.array([query_embedding]).astype('float32'),
        TOP_K_MATCHES
    )
    return [chunks[i] for i in I[0]]

def generate_answer(query: str, context: List[str], model, tokenizer) -> str:
    """Generate answer using TinyLlama"""
    try:
        # Prepare prompt
        context_text = "\n".join(context)
        prompt = f"""<|system|>
You are a helpful assistant. Answer the question based only on the provided context. If the answer cannot be found in the context, say so.

Context:
{context_text}

<|user|>
{query}

<|assistant|>"""
        
        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        # Using max_new_tokens instead of max_length
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Control length of the generated response
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        try:
            response = response.split("<|assistant|>")[-1].strip()
        except:
            response = response.strip()
            
        return response
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None

def main():
    st.title("📚 PDF Question-Answering with RAG")
    st.write("Upload a PDF document and ask questions about its content!")

    # Load models
    with st.spinner("Loading models..."):
        embedding_model, llm_model, tokenizer = load_models()

    # File uploader
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if pdf_file is not None:
        # Check cache for processed embeddings
        cache_path = get_cache_path(pdf_file.name)
        
        if cache_path.exists():
            with cache_path.open('rb') as f:
                cached_data = pickle.load(f)
                chunks = cached_data['chunks']
                faiss_index = cached_data['faiss_index']
        else:
            # Process PDF
            with st.spinner("Processing PDF..."):
                # Extract and preprocess text
                text = extract_text_from_pdf(pdf_file)
                if text:
                    processed_text = preprocess_text(text)
                    chunks = create_chunks(processed_text)
                    
                    # Generate embeddings
                    embeddings = get_embeddings(chunks, embedding_model)
                    if embeddings:
                        # Create FAISS index
                        faiss_index = create_faiss_index(embeddings)
                        
                        # Cache the processed data
                        with cache_path.open('wb') as f:
                            pickle.dump({
                                'chunks': chunks,
                                'faiss_index': faiss_index
                            }, f)
                    else:
                        st.error("Failed to generate embeddings")
                        return
                else:
                    st.error("Failed to extract text from PDF")
                    return

        # Question input
        query = st.text_input("Ask a question about the document:")
        
        if query:
            with st.spinner("Generating answer..."):
                # Get relevant chunks
                relevant_chunks = get_relevant_chunks(query, chunks, faiss_index, embedding_model)
                
                # Generate answer
                answer = generate_answer(query, relevant_chunks, llm_model, tokenizer)
                
                if answer:
                    st.write("### Answer:")
                    st.write(answer)
                    
                    # Show relevant chunks (expandable)
                    with st.expander("View relevant context"):
                        for i, chunk in enumerate(relevant_chunks, 1):
                            st.write(f"**Chunk {i}:**")
                            st.write(chunk)
                            st.write("---")

if __name__ == "__main__":
    # Initialize
    create_cache_dir()
    
    # Set page config
    st.set_page_config(
        page_title="PDF Q&A with RAG",
        page_icon="📚",
        layout="wide"
    )
    
    # Run app
    main()