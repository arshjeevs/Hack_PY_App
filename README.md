# 📚 PDF Chat with RAG

A lightweight **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and **chat with its contents**. Built using `Gradio`, `SentenceTransformers`, `FAISS`, and `TinyLlama`, this tool provides contextual answers to user queries based solely on the PDF content.

---

## ✨ Features

- ✅ Upload and parse any PDF file
- 🧠 Text chunking and embeddings using `SentenceTransformer`
- ⚡ Fast semantic search with `FAISS` vector index
- 💬 Answer generation using `TinyLlama-1.1B-Chat`
- 🖼️ Clean and interactive UI with Gradio
- 💾 Automatic caching of processed PDFs

---

## 🧱 Tech Stack

- **LLM**: [TinyLlama-1.1B-Chat](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Embeddings**: [`all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- **Vector Search**: FAISS
- **PDF Parsing**: `pdfplumber`
- **UI**: Gradio
- **Language**: Python 3.x

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pdf-chat-rag.git
cd pdf-chat-rag
````

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Sample <code>requirements.txt</code></summary>

```txt
torch
faiss-cpu
sentence-transformers
transformers
gradio
pdfplumber
```

</details>

### 4. Run the App

```bash
python app.py
```

The app will be available at: [http://localhost:7860](http://localhost:7860)

---

## 🔍 How It Works

1. **PDF Upload**

   * The user uploads a PDF via Gradio UI.
   * `pdfplumber` extracts raw text from each page.

2. **Preprocessing & Chunking**

   * Text is cleaned and split into overlapping chunks for better context understanding.

3. **Embeddings & FAISS Index**

   * Each chunk is embedded using `all-MiniLM-L12-v2`.
   * Chunks are stored in a `FAISS` vector index for efficient similarity search.

4. **Query Handling**

   * User submits a natural language question.
   * The question is embedded and used to retrieve top-k relevant chunks.

5. **Answer Generation**

   * A structured prompt is formed using the question and retrieved chunks.
   * `TinyLlama-1.1B-Chat` generates an answer grounded in the PDF context.

---

## 💬 Prompt Format

```text
<|system|>
You are a highly detailed and structured assistant...
Context:
[chunk1]
[chunk2]
...

<|user|>
[question]

<|assistant|>
Answer:
1. **Aspect 1: ...**
   - **Definition**: ...
   - **Elaboration**: ...
   - **Example**: ...
   - **Connection**: ...

2. **Aspect 2: ...**
   - ...
```

---

## 🧠 Example Use Case

> Upload a research paper PDF and ask:
>
> * “What are the key contributions of this paper?”
> * “Explain the proposed algorithm.”
> * “Summarize the results in simple terms.”

---

## 🔒 Caching Mechanism

To optimize performance and avoid reprocessing:

* Each PDF is hashed and cached in the `cache/` directory as a `.pkl` file.
* Cached data includes:

  * Preprocessed text chunks
  * FAISS index of embeddings

---

## 📂 Project Structure

```
├── app.py                 # Main application script
├── requirements.txt       # Python dependencies
├── cache/                 # Cached processed PDFs
├── README.md              # This file
```

---

## 🧪 Notes

* If running on CPU, performance may be slower.
* GPU acceleration via CUDA is automatically used if available.
* LLM prompt and answer formatting can be customized in `generate_answer()`.

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

* [HuggingFace Transformers](https://huggingface.co)
* [SentenceTransformers](https://www.sbert.net/)
* [FAISS by Facebook AI](https://github.com/facebookresearch/faiss)
* [Gradio](https://gradio.app)
