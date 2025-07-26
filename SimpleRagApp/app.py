# ===============================
# Imports
# ===============================
import os
import re
import gc
import faiss
import torch
import pickle
import tempfile
import warnings
import traceback
import numpy as np
import pdfplumber
import gradio as gr
from io import BytesIO
from typing import List
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore", category=FutureWarning)

# ===============================
# Configuration
# ===============================
CACHE_DIR = Path("cache")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_DIMENSION = 768
TOP_K_MATCHES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===============================
# Data Structures
# ===============================
@dataclass
class Document:
    text: str
    embedding: np.ndarray = None


# ===============================
# Utilities
# ===============================
def create_cache_dir():
    CACHE_DIR.mkdir(exist_ok=True)


def get_cache_path(filename: str) -> Path:
    return CACHE_DIR / f"{filename}.pkl"


def extract_text_from_pdf(file_obj) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            if hasattr(file_obj, 'name'):
                with open(file_obj.name, 'rb') as f:
                    tmp_file.write(f.read())
            else:
                tmp_file.write(file_obj.read())
            tmp_path = tmp_file.name

        text = ""
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                try:
                    text += page.extract_text() or ""
                except Exception as e:
                    print(f"Error on page: {str(e)}")
                    continue

        os.unlink(tmp_path)

        if not text.strip():
            return "Error: No readable text found in the PDF."

        return text

    except Exception as e:
        if 'tmp_path' in locals():
            os.unlink(tmp_path)
        return f"Error extracting text from PDF: {str(e)}"


def preprocess_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text.strip()


def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    text_length = len(text)

    if text_length < chunk_size:
        return [text] if text else []

    while start < text_length:
        end = start + chunk_size

        if end >= text_length:
            chunks.append(text[start:])
            break

        last_period = text.rfind('.', start, end)
        if last_period != -1 and last_period > start + chunk_size / 2:
            end = last_period + 1
        else:
            while end > start and text[end] != ' ':
                end -= 1
            if end == start:
                end = start + chunk_size

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + chunk_size - overlap, end - overlap)

    return chunks


def get_embeddings(texts: List[str], embedding_model: SentenceTransformer) -> List[np.ndarray]:
    try:
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        return [np.array(embedding) for embedding in embeddings]
    except Exception as e:
        print(f"Error generating embeddings: {str(e)}")
        return None


def create_faiss_index(embeddings_array: List) -> faiss.IndexFlatL2:
    """
    Create a FAISS index from a 2D NumPy array of float32 embeddings.
    
    Parameters:
        embeddings_array (np.ndarray): A 2D NumPy array of shape (n_samples, embedding_dim).
    
    Returns:
        faiss.IndexFlatL2: A FAISS index ready for similarity search.
    """
    embeddings_array = np.array(embeddings_array)

    # Ensure it's a NumPy array
    if not isinstance(embeddings_array, np.ndarray):
        print("Error: embeddings_array is not a numpy ndarray.")
        print(f"Type: {type(embeddings_array)}")
        raise ValueError("Expected embeddings_array to be a numpy ndarray.")

    # Ensure it's 2D
    if embeddings_array.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {embeddings_array.shape}")

    # Ensure float32 type (required by FAISS)
    embeddings_array = embeddings_array.astype('float32')

    # Get embedding dimension
    embedding_dim = embeddings_array.shape[1]

    # Initialize FAISS index
    index = faiss.IndexFlatL2(embedding_dim)

    # Add embeddings to index
    index.add(embeddings_array)

    return index


def get_relevant_chunks(query: str, chunks: List[str], faiss_index: faiss.IndexFlatL2, embedding_model: SentenceTransformer) -> List[str]:
    query_embedding = embedding_model.encode([query])[0]
    D, I = faiss_index.search(
        np.array([query_embedding]).astype('float32'),
        min(TOP_K_MATCHES, len(chunks))
    )
    return [chunks[i] for i in I[0]]


def generate_answer(query: str, context: List[str], model, tokenizer) -> str:
    try:
        context_text = "\n".join(context)
        prompt = f"""<|system|>
You are a highly detailed and structured assistant...
Context:
{context_text}

<|user|>
{query}

<|assistant|>
Answer:
1. **Aspect 1: Title/Topic**  
   - **Definition**: Provide a detailed definition of the aspect...  
   - **Elaboration**: Explain its significance...  
   - **Example**: If relevant...  
   - **Connection**: Describe connections...

2. **Aspect 2: Title/Topic**  
   ...
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            num_return_sequences=1,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("<|assistant|>")[-1].strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def load_models():
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2', device=DEVICE)
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    return embedding_model, model, tokenizer


# ===============================
# Main Class
# ===============================
class PDFQuestionAnswering:
    def __init__(self):
        create_cache_dir()
        print(f"Using device: {DEVICE}")
        self.embedding_model, self.llm_model, self.tokenizer = load_models()
        self.current_chunks = None
        self.current_faiss_index = None
        self.chat_history = []

    def process_pdf(self, pdf_file):
        if pdf_file is None:
            return [], "Please upload a PDF file."

        try:
            cache_path = get_cache_path(pdf_file.name)
            if cache_path.exists():
                print("Loading from cache...")
                with cache_path.open('rb') as f:
                    cached_data = pickle.load(f)
                    self.current_chunks = cached_data['chunks']
                    self.current_faiss_index = cached_data['faiss_index']
                return [], "PDF loaded from cache! You can now ask questions about the document."

            print("Processing new PDF...")
            text = extract_text_from_pdf(pdf_file)
            if isinstance(text, str) and not text.startswith("Error"):
                processed_text = preprocess_text(text)
                if not processed_text:
                    return [], "No readable text found in the PDF."

                self.current_chunks = create_chunks(processed_text)
                if not self.current_chunks:
                    return [], "No valid text chunks could be created from the PDF."

                embeddings = get_embeddings(self.current_chunks, self.embedding_model)
                if embeddings:
                    self.current_faiss_index = create_faiss_index(embeddings)
                    with cache_path.open('wb') as f:
                        pickle.dump({
                            'chunks': self.current_chunks,
                            'faiss_index': self.current_faiss_index
                        }, f)
                    return [], f"PDF processed successfully! You can now ask questions about the document."
                else:
                    return [], "Failed to generate embeddings"
            else:
                return [], text
        except Exception as e:
            traceback.print_exc()
            return [], f"Error processing PDF: {str(e)}"

    def chat(self, message, history):
        if self.current_chunks is None or self.current_faiss_index is None:
            return "Please upload and process a PDF first."
        if not message.strip():
            return "Please enter a question."
        try:
            relevant_chunks = get_relevant_chunks(
                message, self.current_chunks,
                self.current_faiss_index, self.embedding_model
            )

            if not relevant_chunks:
                return "I couldn't find relevant information in the document to answer your question."

            answer = generate_answer(
                message, relevant_chunks, self.llm_model, self.tokenizer
            )

            source_context = "\n\n🔍 *Source Context*:\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                source_context += f"\n{chunk}\n---"

            return answer
        except Exception as e:
            return f"Error answering question: {str(e)}"


# ===============================
# Gradio UI
# ===============================
def create_gradio_interface():
    qa_system = PDFQuestionAnswering()

    with gr.Blocks(title="PDF Chat with RAG") as interface:
        gr.Markdown("# 📚 Chat with your PDF")
        gr.Markdown("Upload a PDF document and start a conversation about its content!")

        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                process_button = gr.Button("Process PDF")
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=450, type="messages")
                message = gr.Textbox(
                    label="Ask a question about the document",
                    placeholder="Type your question here...",
                    lines=2
                )
                with gr.Row():
                    submit = gr.Button("Send")
                    clear = gr.Button("Clear Chat")

        def respond(message, history):
            bot_message = qa_system.chat(message, history)
            history = history or []
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": bot_message})
            return "", history


        def clear_chat():
            return None, None

        process_button.click(
            fn=qa_system.process_pdf,
            inputs=[pdf_input],
            outputs=[chatbot, message]
        )

        submit.click(fn=respond, inputs=[message, chatbot], outputs=[message, chatbot])
        message.submit(fn=respond, inputs=[message, chatbot], outputs=[message, chatbot])
        clear.click(fn=clear_chat, outputs=[message, chatbot])

    return interface


# ===============================
# Run the App
# ===============================
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
