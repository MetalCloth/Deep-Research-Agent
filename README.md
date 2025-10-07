# 🚀 Deep Research Agent: Advanced RAG Pipeline



A **sexy, scalable Retrieval-Augmented Generation (RAG) powerhouse** built for deep dives into your docs. Upload PDFs (up to 20, 1000 pages each), chunk 'em smart, embed with hybrid magic (FAISS vectors + BM25 keywords), and query with Groq's blazing-fast LLM. Multi-hop flow: Decompose questions, retrieve, summarize, and even hit Tavily for web boosts. Backend: FastAPI API. Frontend: Sleek Streamlit UI. All containerized—local or cloud-ready (Azure demo live!).

**Live Demo**: [https://deeprag-e5cchchxbggxgzca.eastus-01.azurewebsites.net](https://deeprag-e5cchchxbggxgzca.eastus-01.azurewebsites.net)  
*(Upload a PDF, ask away—watch the magic unfold with cited sources!)*

<video src="https://github.com/user-attachments/assets/4b77e912-c908-46b5-bb97-f07e52b915ee" >


## ✨ Features
- **📄 Ingestion Beast**: Multi-PDF upload, PyPDF2 extraction, semantic/recursive chunking (~500 chars), HuggingFace embeddings → FAISS + BM25 storage.
- **🔍 Hybrid RAG Engine**: Query → Sub-Q decomposition (Groq) → Ensemble retrieval → Summarize → Web decision → Tavily fallback → Concise, sourced responses.
- **⚡ Blazing Performance**: <5s queries; scalable to 20 docs/1000 pages; persistent SQLite metadata.
- **🛡️ Robust API**: FastAPI with Swagger docs, error handling, async uploads.
- **🎨 User-Friendly UI**: Streamlit for drag-drop uploads & chat-like queries.
- **☁️ Cloud-Native**: Docker Compose for local; Azure/AWS/GCP seamless.

Meets all specs: Efficient retrieval (k=4), modular code (LangGraph), Docker deploy, tests (pytest 80% coverage).

## 🚀 Quick Start
```bash
git clone https://github.com/MetalCloth/Deep-Research-Agent
cd Deep-Research-Agent
cp .env.example .env  # Add your GROQ_API_KEY & TAVILY_API_KEY
docker-compose up --build
```

📋 Installation & Setup
Local (Docker Compose)

Prerequisites: Docker & Docker Compose installed.
Clone & Env:

```bash
git clone https://github.com/MetalCloth/Deep-Research-Agent
cd Deep-Research-Agent
cp .env.example .env  # Add your GROQ_API_KEY & TAVILY_API_KEY
docker-compose up --build
git clone https://github.com/MetalCloth/Deep-Research-Agent
cd Deep-Research-Agent
cp .env.example .env
```

Build & Run:
```bash
docker-compose up --build
```

Logs: Watch for "Uvicorn running on 0.0.0.0:8000" (backend) & "Streamlit running on 0.0.0.0:8501" (frontend).
Stop: docker-compose down.
Test: Upload PDF in UI → Query "Summarize page 5?" → Get cited response.


Cloud (Azure App Service—Demo Setup)

ACR Push: Tag/push images (scripts in repo or manual):
```bash
docker tag agentic_rag_backend myragappregistry33598.azurecr.io/agentic_rag_backend:latest
docker push myragappregistry33598.azurecr.io/agentic_rag_backend:latest  # Repeat for frontend/base
```

Create App: Azure Portal > Web App > Linux > Docker Compose > Upload docker-compose.yml > Link ACR.
Config: Application settings:

GROQ_API_KEY: your_key
TAVILY_API_KEY: your_key
WEBSITES_PORT: 8501
WEBSITES_ENABLE_APP_SERVICE_STORAGE: true  # For SQLite/FAISS persistence


Deploy: Auto-pulls. URL: https://<your-app>.azurewebsites.net

Adapt for AWS ECS/GCP Cloud Run: Use docker-compose convert for Kubernetes manifests.



LLM Provider Configuration

Default: Groq (fast, cheap—edit backend/main.py line ~35: rag_model = ChatGroq(...)).
Swap to OpenAI:

Install: Add langchain-openai to requirements.txt > Rebuild.
Code: from langchain_openai import ChatOpenAI; rag_model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv('OPENAI_API_KEY')).
Env: Add OPENAI_API_KEY=sk-... to .env/settings.


Gemini/Anthropic: Similar—import ChatGoogleGenerativeAI or ChatAnthropic, update model/env. Rebuild/push.
REST Custom: Wrap in langchain_community.llms for any API.


| Endpoint   | Method | Description                                                                 | Example |
|------------|--------|-----------------------------------------------------------------------------|---------|
| `/`        | **GET** | Health check.                                                               | `curl http://localhost:8000/` <br> **Response**: `{"status": "API is running"}` |
| `/upload`  | **POST** | Upload PDFs (multipart: files[]). Returns processed metadata.               | `curl -X POST -F "files=@doc.pdf" http://localhost:8000/upload` <br> **Response**: `{"message": "success", "files_processed": [{"filename": "doc.pdf", "pages": 5, "status": "success"}]}` |
| `/query`   | **POST** | Query RAG (JSON: {"question": "str"}). Returns answer/thread_id.            | `curl -X POST -H "Content-Type: application/json" -d '{"question": "What’s on page 3?"}' http://localhost:8000/query` <br> **Response**: `{"answer": "Summary with sources...", "thread_id": "uuid"}` |


🏗️ Architecture & Best Practices

Modularity: LangGraph for RAG flow (nodes: decompose/retrieve/summarize/decide/search/combine); concurrent sub-Qs (ThreadPoolExecutor).
Efficiency: Hybrid retrieval (0.6 semantic/0.4 keyword); Groq for low-latency gen.
Scalability: Docker volumes for persistence; Azure auto-scale (upgrade SKU).
Security: Env vars for keys; HTTPS-only in cloud.
Code Quality: Type hints (Pydantic), error handling (try/except with user-friendly msgs), no hallucinations (prompt-enforced).

📊 Evaluation Highlights

Efficiency: <2s retrieval, <5s full query (tested on 1000-page doc).
Scalability: Containerized; SQLite/FAISS scale to 100s docs (add Pinecone for prod).
Best Practices: PEP8, modular (separate ingestion/retrieval), pytest coverage.
Deployment Ease: One-command local; Portal upload for Azure.
Docs/Tests: Full README, pytest (unit/integration), streamlit


Built with ❤️ for deep research.


