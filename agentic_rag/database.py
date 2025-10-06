import os
import pickle
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- Configuration ---
FAISS_PATH = "vectorstore"
BM25_PATH = "bm25.pkl"
EMBEDDING_MODEL = "snowflake-arctic-embed"

# --- Models and Splitters (Initialized once) ---
embedding_model = OllamaEmbeddings(model=EMBEDDING_MODEL)

semantic_text_splitter = SemanticChunker(
    embedding_model,
    min_chunk_size=500,
    buffer_size=100
)

keyword_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

def ingest(pdf_file_like_object):
    """
    Ingests a PDF, processes it, and updates the vector stores.
    This function is UI-independent and returns a status dictionary.
    
    Args:
        pdf_file_like_object: A file-like object (e.g., from FastAPI's UploadFile).
                              It must have a '.name' attribute.
    """
    pdf_name = getattr(pdf_file_like_object, 'name', 'unknown.pdf')
    print(f"--- Starting ingestion for: {pdf_name} ---")

    reader = PdfReader(pdf_file_like_object)
    doc_pages = []
    for page_num, page in enumerate(reader.pages):
        content = page.extract_text() or ""
        doc = Document(
            page_content=content,
            metadata={"source": pdf_name, "page": page_num + 1}
        )
        doc_pages.append(doc)

    print(f"Ingested {len(doc_pages)} pages from {pdf_name}.")

    # --- Chunking ---
    print("Splitting documents into chunks...")
    semantic_chunks = semantic_text_splitter.split_documents(doc_pages)
    keyword_chunks = keyword_text_splitter.split_documents(doc_pages)
    print(f"Created {len(semantic_chunks)} semantic and {len(keyword_chunks)} keyword chunks.")

    # --- Vector Store Processing ---
    print("Updating FAISS vector store...")
    if os.path.exists(FAISS_PATH):
        faiss_store = FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        faiss_store.add_documents(semantic_chunks)
    else:
        faiss_store = FAISS.from_documents(semantic_chunks, embedding_model)
    faiss_store.save_local(FAISS_PATH)
    print(f"FAISS vector store saved to '{FAISS_PATH}'.")

    # --- BM25 Retriever Processing ---
    print("Updating BM25 retriever...")
    all_keyword_docs = keyword_chunks
    if os.path.exists(BM25_PATH):
        with open(BM25_PATH, "rb") as f:
            existing_bm25 = pickle.load(f)
            all_keyword_docs.extend(existing_bm25.docs)
    
    bm25 = BM25Retriever.from_documents(all_keyword_docs)
    bm25.k = 4
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 retriever saved to '{BM25_PATH}'.")
    
    # --- Return Status ---
    print(f"--- Finished ingestion for: {pdf_name} ---")
    return {"filename": pdf_name, "pages": len(doc_pages), "status": "success"}