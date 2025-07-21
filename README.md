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
    A["User Question"] --> B["Subquestion Decomposition (LLM)"]

    B --> C1["SubQ1"]
    B --> C2["SubQ2"]
    B --> C3["SubQn"]

    %% SubQ1 Path
    C1 --> D1["Local Retrieval (FAISS + BM25)"]
    D1 --> R1["Doc Context 1"]
    C1 -->|If Needed| W1["Tavily Web Search"]
    W1 --> T1["Web Result 1"]
    R1 --> M1["Merged Context 1"]
    T1 --> M1
    M1 --> S1["SubAnswer 1"]

    %% SubQ2 Path
    C2 --> D2["Local Retrieval (FAISS + BM25)"]
    D2 --> R2["Doc Context 2"]
    C2 -->|If Needed| W2["Tavily Web Search"]
    W2 --> T2["Web Result 2"]
    R2 --> M2["Merged Context 2"]
    T2 --> M2
    M2 --> S2["SubAnswer 2"]

    %% SubQn Path
    C3 --> D3["Local Retrieval (FAISS + BM25)"]
    D3 --> R3["Doc Context 3"]
    C3 -->|If Needed| W3["Tavily Web Search"]
    W3 --> T3["Web Result 3"]
    R3 --> M3["Merged Context 3"]
    T3 --> M3
    M3 --> S3["SubAnswer 3"]

    %% Final Synthesis
    S1 --> F["Final Synthesizer"]
    S2 --> F
    S3 --> F
    F --> Z["Final Answer"]
```




---

## 🧠 Key Features

- 🔗 **LangGraph Multi-Agent Loop**: Pro/Con/Validator style agents
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

