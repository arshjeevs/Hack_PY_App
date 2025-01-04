import gradio as gr
import numpy as np
from typing import List, Tuple
import faiss
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import re
from dataclasses import dataclass
import pickle
import os
from pathlib import Path
from io import BytesIO
import pdfplumber
import tempfile

# Configuration
CACHE_DIR = Path("cache")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBED_DIMENSION = 768
TOP_K_MATCHES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Document:
    text: str
    embedding: np.ndarray = None

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
        if last_period != -1 and last_period > start + chunk_size/2:
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

def create_faiss_index(embeddings: List[np.ndarray]) -> faiss.IndexFlatL2:
    embeddings_array = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(EMBED_DIMENSION)
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
You are a highly detailed and structured assistant. For each question, you will provide a comprehensive, in-depth explanation with the following characteristics:
1. Break down the answer into numbered or bullet points, covering each distinct aspect of the question.
2. For each aspect, provide a detailed explanation, including definitions, relevant examples, and any further elaboration that adds clarity and depth.
3. If applicable, explain the connections between different aspects or how they contribute to a larger concept or framework.
4. If the answer cannot be found in the provided context, explicitly say "I don't know."

Context:
{context_text}

<|user|>
{query}

<|assistant|>
Answer:
1. **Aspect 1: Title/Topic**  
   - **Definition**: Provide a detailed definition of the aspect based on the context. 
   - **Elaboration**: Explain its significance and the role it plays in the context.  
   - **Example**: If relevant, provide a specific example that illustrates this aspect.  
   - **Connection**: Describe how this aspect connects with other aspects or ideas in the context.

2. **Aspect 2: Title/Topic**  
   - **Definition**: Give a clear, detailed definition of this aspect.  
   - **Elaboration**: Expand on its importance and any complexities associated with it.  
   - **Example**: If possible, offer a real-world or hypothetical example.  
   - **Connection**: Discuss any relationships with the previous or following aspects.  

3. **Aspect 3: Title/Topic**  
   - **Definition**: Thoroughly explain the meaning of this aspect.  
   - **Elaboration**: Provide insights into why this is a key part of the concept.  
   - **Example**: Use specific examples or case studies that apply this aspect in practice.  
   - **Connection**: Show how this aspect interrelates with others.

[Continue in the same manner for all other aspects]
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
        try:
            response = response.split("<|assistant|>")[-1].strip()
        except:
            response = response.strip()

        return response
    except Exception as e:
        return f"Error generating answer: {str(e)}"

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
            return [], f"Error processing PDF: {str(e)}"

    def chat(self, message, history):
        if self.current_chunks is None or self.current_faiss_index is None:
            return "Please upload and process a PDF first."

        if not message.strip():
            return "Please enter a question."

        try:
            relevant_chunks = get_relevant_chunks(
                message,
                self.current_chunks,
                self.current_faiss_index,
                self.embedding_model
            )

            if not relevant_chunks:
                return "I couldn't find relevant information in the document to answer your question."

            answer = generate_answer(
                message,
                relevant_chunks,
                self.llm_model,
                self.tokenizer
            )

            source_context = "\n\n🔍 *Source Context*:\n"
            for i, chunk in enumerate(relevant_chunks, 1):
                source_context += f"\n{chunk}\n---"

            return answer + source_context
        except Exception as e:
            return f"Error answering question: {str(e)}"

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
                chatbot = gr.Chatbot(height=450)
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
            history.append((message, bot_message))
            return "", history

        def clear_chat():
            return None, None

        process_button.click(
            fn=qa_system.process_pdf,
            inputs=[pdf_input],
            outputs=[chatbot, message]
        )

        submit.click(
            fn=respond,
            inputs=[message, chatbot],
            outputs=[message, chatbot]
        )
        message.submit(
            fn=respond,
            inputs=[message, chatbot],
            outputs=[message, chatbot]
        )
        clear.click(
            fn=clear_chat,
            outputs=[message, chatbot],
        )

    return interface

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True)