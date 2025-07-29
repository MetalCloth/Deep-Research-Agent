# 🔍 Multi-Hop Hybrid RAG with LangGraph + Tavily + Claude (Agentic QA System)

This project implements an **Agentic Multi-Hop RAG (Retrieval-Augmented Generation)** system using:

- 🧠 **LangGraph** for multi-agent orchestration
- 📄 **Hybrid Retrieval** via FAISS (Ollama embeddings) + BM25
- 🌐 **Tavily Web Search** (conditionally used)
- 🤖 **Claude 3.5 Haiku** (via Anthropic) as the main LLM or can use **GROQ Agents**
- 📚 **SemanticChunker** + PDF loader + Metadata tracking
- 🧪 **LangChain Expression Language** for modularity

---

## ⚙️ How it works

```mermaid
graph TD
    %% Ingestion Pipeline
    A1[PDF Upload] --> A2[Parse and Split Pages]
    A2 --> A3[Generate Embeddings using Ollama]
    A2 --> A4[Build BM25 Index]
    A3 --> A5[Store in FAISS Vector DB]
    A4 --> A6[BM25 Retriever]
    A5 --> A7[FAISS Retriever]
    A6 --> A8[Hybrid Retriever]
    A7 --> A8

    %% Question Handling
    A8 --> B1[User Question]
    B1 --> B2[Decompose into Sub-Questions]

    %% Sub-Question 1
    B2 --> C1[SubQ1]
    C1 --> D1[Retrieve Local Docs FAISS and BM25]
    D1 --> E1[Local Context 1]
    C1 -->|If Needed| F1[Web Search via Tavily]
    F1 --> G1[Web Context 1]
    E1 --> H1[Merge Contexts for SubQ1]
    G1 --> H1
    H1 --> I1[Generate Answer for SubQ1]

    %% Sub-Question 2
    B2 --> C2[SubQ2]
    C2 --> D2[Retrieve Local Docs]
    D2 --> E2[Local Context 2]
    C2 -->|If Needed| F2[Web Search]
    F2 --> G2[Web Context 2]
    E2 --> H2[Merge Contexts for SubQ2]
    G2 --> H2
    H2 --> I2[Generate Answer for SubQ2]

    %% Sub-Question N
    B2 --> C3[SubQn]
    C3 --> D3[Retrieve Local Docs]
    D3 --> E3[Local Context 3]
    C3 -->|If Needed| F3[Web Search]
    F3 --> G3[Web Context 3]
    E3 --> H3[Merge Contexts for SubQn]
    G3 --> H3
    H3 --> I3[Generate Answer for SubQn]

    %% Final Synthesis
    I1 --> Z1[Synthesize Final Answer]
    I2 --> Z1
    I3 --> Z1
    Z1 --> Z2[Final Answer to User]

```




---

## Demo
https://github.com/user-attachments/assets/1e07b283-ad23-4cf1-8a9e-10854ea513f0

---



## 🧠 Key Features

- 📘 **PDF Upload + Semantic Chunking** for intelligent context
- 🔄 **Hybrid Retrieval**: FAISS for embeddings + BM25 for keyword match
- 🌐 **Optional Web Search** using Tavily only when needed
- 🧩 **Context Merging** for sub-question aggregation
- 🧵 **Subanswer Synthesis** for structured final output

---

## 🛠️ Tech Stack

- `LangChain`, `LangGraph`, `Anthropic`, `Tavily`
- `FAISS`, `BM25Retriever`, `OllamaEmbeddings`
- `PyMuPDFLoader`, `SemanticChunker`
- `Streamlit` for UI (optional)

---

## 🚀 Getting Started

1. Clone the repo  
   ```bash
   git clone https://github.com/yourname/hybrid-multihop-rag.git
   cd hybrid-multihop-rag
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Add your API keys:
   - `.env` file or directly in code
     ```
     ANTHROPIC_API_KEY=...
     TAVILY_API_KEY=...
     ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## ✍️ Example Prompt

> **Q:** What are the recent climate policy changes in the EU, and how do they compare to US initiatives?

- 🔹 SubQs:
  - What are recent climate policy changes in the EU?
  - What are similar US climate policy updates?
  - How do the two compare?

Final synthesized answer is returned after merging relevant document + web search results.

---

## 🤖 Credits

Made with ❤️ by [Puneet Rawat] using LangGraph + Claude/Groq + Tavily + LangChain + FAISS magic.

