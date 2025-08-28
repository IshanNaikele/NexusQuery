from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pinecone import Pinecone
from pydantic import BaseModel
from typing import List, Optional
import tempfile
import shutil
from dotenv import load_dotenv
import os

from backend.ingest.pdf import process_and_ingest_pdf
from backend.data_retriever.query import get_answer_from_docs

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)

class QueryRequest(BaseModel):
    query: str
    chat_history: List[tuple] = []

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model="llama3-70b-8192"
)

# Global variable to store documents
document_store: Optional[List[str]] = None

app = FastAPI(title="NexusQuery Project")

@app.post("/ingest/pdf")
async def ingest_pdf_endpoint(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        result, processed_docs = process_and_ingest_pdf(temp_file_path, pc, embeddings)
        global document_store
        document_store = [doc.page_content for doc in processed_docs]
        return result
    finally:
        os.remove(temp_file_path)

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    if document_store is None or len(document_store) == 0:
        raise HTTPException(status_code=400, detail="No documents have been ingested yet.")
    
    answer = get_answer_from_docs(
        request.query, 
        request.chat_history, 
        pc, 
        embeddings, 
        llm, 
        document_store=document_store   
    )
    return {"query": request.query, "answer": answer}