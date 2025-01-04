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
import time
import psutil
from datetime import datetime

# Configuration
CACHE_DIR = Path("cache")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_DIMENSION = 512  # Dimension for MiniLM embeddings
TOP_K_MATCHES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models
@st.cache_resource
def load_models():
    # Load embedding model
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=DEVICE
    )

    # Load LLM and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(DEVICE)

    return embedding_model, model, tokenizer

def log_performance_metrics():
    """Log system and GPU performance metrics."""
    # System memory usage
    memory = psutil.virtual_memory()
    system_memory_usage = memory.percent

    # GPU memory usage (if CUDA is available)
    gpu_memory_usage = None
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        gpu_memory_usage = f"{gpu_memory:.2f} GB"
        gpu_utilization = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_info = f"{gpu_utilization:.2f} GB"
    else:
        gpu_memory_usage = "GPU unavailable"

    # Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Log the metrics
    st.sidebar.markdown("### Performance Metrics")
    st.sidebar.markdown(f"**Timestamp:** {timestamp}")
    st.sidebar.markdown(f"**System Memory Usage:** {system_memory_usage}%")
    if gpu_memory_usage != "GPU unavailable":
        st.sidebar.markdown(f"**GPU Memory Usage:** {gpu_memory_usage}")
        st.sidebar.markdown(f"**Total GPU Capacity**: {gpu_info}")
    else:
        st.sidebar.markdown("**GPU Memory Usage:** GPU unavailable")

def profile_execution(func):
    """Decorator to profile function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        st.sidebar.markdown(f"**Execution Time ({func.__name__}):** {execution_time:.4f} seconds")
        return result
    return wrapper

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

@profile_execution
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


import re


def preprocess_text(text: str) -> str:
    """Clean and preprocess extracted text"""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?-]", "", text)
    return text.strip()


def create_chunks(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end < text_length:
            while end > start and text[end] != " ":
                end -= 1
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end - overlap

    return chunks

@profile_execution
def get_embeddings(
    texts: List[str], embedding_model: SentenceTransformer
) -> List[np.ndarray]:
    """Generate embeddings using sentence-transformers"""
    try:
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        return [np.array(embedding) for embedding in embeddings]
    except Exception as e:
        st.error(f"Error generating embeddings: {str(e)}")
        return None

@profile_execution
def create_faiss_index(embeddings: List[np.ndarray]) -> faiss.IndexFlatL2:
    """Create and populate FAISS index"""
    embeddings_array = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(EMBED_DIMENSION)
    index.add(embeddings_array)
    return index

@profile_execution
def get_relevant_chunks(
    query: str,
    chunks: List[str],
    faiss_index: faiss.IndexFlatL2,
    embedding_model: SentenceTransformer,
) -> List[str]:
    """Retrieve the most relevant chunks for the query using FAISS search"""
    try:
        # Get embedding for the query
        query_embedding = embedding_model.encode([query])[0]

        # Perform FAISS search
        D, I = faiss_index.search(
            np.array([query_embedding]).astype("float32"),
            TOP_K_MATCHES,  # Define how many relevant chunks you want
        )

        # Retrieve and return the relevant chunks
        relevant_chunks = [chunks[i] for i in I[0]]
        return relevant_chunks
    except Exception as e:
        st.error(f"Error retrieving relevant chunks: {str(e)}")
        return []


@profile_execution
def generate_answer(query: str, context: List[str], model, tokenizer) -> str:
    """Generate a detailed and structured answer with improved context."""
    try:
        # Join the context into a longer text
        context_text = "\n\n".join(
            [f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(context)]
        )

        # Create the refined and detailed prompt
        prompt = f"""
<|system|>
You are a knowledgeable assistant specializing in answering questions based on provided documents. Follow these instructions to generate comprehensive, detailed answers:

1. **Introduction**:
   - Start with a brief overview of the key topic or concept from the provided context that is relevant to the query. Be sure to cover all important facets of the subject.

2. **Detailed Explanation**:
   - Break down the answer into numbered points for clarity, and elaborate on each point thoroughly using information from the provided context. 
   - Focus on in-depth explanations and include as many details as possible, supporting your points with relevant quotes or references from the context.

3. **Conclusion**:
   - Provide a final summary based on the context. Offer a strong conclusion that ties together the key points and addresses the query clearly.

4. **Important Notes**:
   - Base your answer ONLY on the information provided in the context below.
   - If the complete answer is not available in the context, explicitly state: 
     "Based on the provided context, I cannot fully answer this question."
   - Avoid making unsupported assumptions or adding external information.

Context:
{context_text}

<|user|>
{query}

<|assistant|>
Let me provide a structured answer based on the provided context:
"""

        # Generate the response
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,  # Increased for more detailed output
            num_return_sequences=1,
            temperature=0.7,  # Adjusted for more flexibility in response
            do_sample=True,  # Disable sampling for more focused answers
            top_p=0.95,  # Slightly higher for more diversity
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode and format the response
        raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract and return the structured part of the response
        try:
            if (
                "Let me provide a structured answer based on the provided context:"
                in raw_response
            ):
                structured_part = raw_response.split(
                    "Let me provide a structured answer based on the provided context:"
                )[-1].strip()
                return structured_part
            return raw_response.strip()
        except Exception as e:
            st.error(f"Error formatting response: {str(e)}")
            return raw_response
    except Exception as e:
        st.error(f"Error generating answer: {str(e)}")
        return None


def main():
    
    device_status = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    st.sidebar.info(f"**Device in use:** {device_status}")
    
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
            with cache_path.open("rb") as f:
                cached_data = pickle.load(f)
                chunks = cached_data["chunks"]
                faiss_index = cached_data["faiss_index"]
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
                        with cache_path.open("wb") as f:
                            pickle.dump(
                                {"chunks": chunks, "faiss_index": faiss_index}, f
                            )
                    else:
                        st.error("Failed to generate embeddings")
                        return
                else:
                    st.error("Failed to extract text from PDF")
                    return

        # Question input
        query = st.text_input("Ask a question about the document:")
        start_time = time.time()
        if query:
            with st.spinner("Generating answer..."):
                # Get relevant chunks
                relevant_chunks = get_relevant_chunks(
                    query, chunks, faiss_index, embedding_model
                )

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
            end_time = time.time()
            processing_time = end_time - start_time
            st.write(f"{processing_time:.6f} seconds")


if __name__ == "__main__":
    # Initialize
    create_cache_dir()

    # Set page config
    st.set_page_config(page_title="PDF Q&A with RAG", page_icon="📚", layout="wide")

    # Run app
    main()
