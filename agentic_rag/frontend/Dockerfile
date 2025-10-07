FROM myragappregistry33598.azurecr.io/agentic_rag_base:latest


COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.port", "8501", "--server.address", "0.0.0.0"]