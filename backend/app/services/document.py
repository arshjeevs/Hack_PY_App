import fitz  # PyMuPDF

# Extract text from PDF binary and split into overlapping chunks
def extract_text_and_chunk(pdf_bytes: bytes, chunk_size=300, overlap=50):
    pdf_doc = fitz.open("pdf", pdf_bytes)
    full_text = ""

    for page in pdf_doc:
        full_text += page.get_text()

    pdf_doc.close()

    # Simple chunking
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
