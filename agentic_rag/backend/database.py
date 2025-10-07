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
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

FAISS_PATH = "vectorstore"
BM25_PATH = "bm25.pkl"
EMBEDDING_MODEL = "snowflake-arctic-embed"


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

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
    pdf_name = getattr(pdf_file_like_object, 'name', 'unknown.pdf')

    reader = PdfReader(pdf_file_like_object)
    doc_pages = []
    for page_num, page in enumerate(reader.pages):
        content = page.extract_text() or ""
        doc = Document(
            page_content=content,
            metadata={"source": pdf_name, "page": page_num + 1}
        )
        doc_pages.append(doc)


    semantic_chunks = semantic_text_splitter.split_documents(doc_pages)
    keyword_chunks = keyword_text_splitter.split_documents(doc_pages)

    faiss_index_path = os.path.join(FAISS_PATH, "index.faiss")
    os.makedirs(FAISS_PATH, exist_ok=True)

    if os.path.exists(faiss_index_path):
        try:
            faiss_store = FAISS.load_local(FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            faiss_store.add_documents(semantic_chunks)
        except Exception as e:
            print(e)
            faiss_store = FAISS.from_documents(semantic_chunks, embedding_model)
    else:
        faiss_store = FAISS.from_documents(semantic_chunks, embedding_model)

    faiss_store.save_local(FAISS_PATH)

    print("Updating BM25 retriever...")
    all_keyword_docs = keyword_chunks
    if os.path.exists(BM25_PATH):
        try:
            with open(BM25_PATH, "rb") as f:
                existing_bm25 = pickle.load(f)
                all_keyword_docs.extend(existing_bm25.docs)
        except Exception as e:
            print(e)

    bm25 = BM25Retriever.from_documents(all_keyword_docs)
    bm25.k = 4
    with open(BM25_PATH, "wb") as f:
        pickle.dump(bm25, f)

    return {"filename": pdf_name, "pages": len(doc_pages), "status": "success"}
