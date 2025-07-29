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
import streamlit as st

load_dotenv()
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY')

# Embeddings and chunkers
embedding_model = OllamaEmbeddings(model='snowflake-arctic-embed')

semantic_text_splitter = SemanticChunker(
    embedding_model,
    min_chunk_size=500,
    buffer_size=100
)

keyword_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

if "pdf_texts" not in st.session_state:
    st.session_state.pdf_texts = {} 
if "doc_store" not in st.session_state:
    st.session_state.doc_store = []  


def ingest(pdf,vectorstore_path: str = "vectorstore",bm25_path:str="bm25.pkl"):
    if pdf.name in st.session_state.pdf_texts:
        return

    reader = PdfReader(pdf)
    full_text = ""
    page_number = 1
    doc_pages = []

    for page in reader.pages:
        content = page.extract_text() or ""
        full_text += content
        doc = Document(
            page_content=content,
            metadata={"source": pdf.name, "page": page_number}
        )
        st.session_state.doc_store.append(doc)
        doc_pages.append(doc)
        page_number += 1

    st.session_state.pdf_texts[pdf.name] = full_text
    st.success(f"Ingested '{pdf.name}' ({len(doc_pages)} pages)")

    with st.spinner(f"Splitting '{pdf.name}' into chunks..."):
        
        semantic_chunks = semantic_text_splitter.split_documents(doc_pages)
        keyword_chunks = keyword_text_splitter.split_documents(doc_pages)

    with st.spinner("Saving vectorstore and BM25..."):
        embeddings = OllamaEmbeddings(model="snowflake-arctic-embed:latest")

        if os.path.exists(vectorstore_path):
            faiss_store = FAISS.load_local(vectorstore_path, embeddings,allow_dangerous_deserialization=True)
            faiss_store.add_documents(semantic_chunks)
        else:
            faiss_store = FAISS.from_documents(semantic_chunks, embeddings)

        faiss_store.save_local(vectorstore_path)

        if os.path.exists(bm25_path):
            with open(bm25_path, "rb") as f:
                bm25 = pickle.load(f)
                final_docs = bm25.docs+keyword_chunks
        else:
            final_docs = keyword_chunks

        bm25 = BM25Retriever.from_documents(final_docs)
        bm25.k = 4

        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)

    st.success(f" Saved FAISS: {vectorstore_path}")
    st.success(f" Saved BM25: {bm25_path}")

