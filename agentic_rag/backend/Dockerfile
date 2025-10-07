FROM myragappregistry33598.azurecr.io/agentic_rag_base:latest


COPY . .

EXPOSE 8000

CMD ["uvicorn", "ai:api", "--host", "0.0.0.0", "--port", "8000"]