# api.py
import io
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

# Import the refactored, UI-independent components
from asscheeks import agent_app
from database import ingest  # Assuming database.py is also UI-free

# --- FastAPI App Initialization ---
api = FastAPI(
    title="DeepTrace RAG API",
    description="API for uploading documents and querying a multi-source RAG system.",
    version="1.0.0"
)

# --- Pydantic Models for API ---
class QueryRequest(BaseModel):
    question: str
    thread_id: Optional[str] = str(uuid.uuid4())

class QueryResponse(BaseModel):
    answer: str
    thread_id: str
    
class UploadResponse(BaseModel):
    message: str
    files_processed: List[dict]

# --- API Endpoints ---
@api.post("/upload", response_model=UploadResponse, summary="Upload PDF documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    processed_files = []
    for file in files:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail=f"File '{file.filename}' is not a PDF.")
        
        try:
            contents = await file.read()
            pdf_file_obj = io.BytesIO(contents)
            pdf_file_obj.name = file.filename
            
            result = ingest(pdf_file_obj)
            processed_files.append(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing '{file.filename}': {str(e)}")
    
    return UploadResponse(message="Upload completed successfully.", files_processed=processed_files)

@api.post("/query", response_model=QueryResponse, summary="Query the RAG system")
def query_system(request: QueryRequest):
    thread_id = request.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    graph_input = {"question": request.question, "messages": [HumanMessage(content=request.question)]}
    
    try:
        response = agent_app.invoke(graph_input, config=config)
        final_answer = response.get('final_answer', "Error: Could not retrieve a final answer.")
        return QueryResponse(answer=final_answer, thread_id=thread_id)
    except FileNotFoundError as e:
         raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during query processing: {str(e)}")

@api.get("/", summary="Health Check")
def read_root():
    return {"status": "API is running"}