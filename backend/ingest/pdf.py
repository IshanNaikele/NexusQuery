from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone,ServerlessSpec
from typing import Any
from fastapi import HTTPException
from dotenv import load_dotenv
load_dotenv()
index_name = "nexus-query-index"
 
 
embedding_dim = 768


def process_and_ingest_pdf(file_path:str,pc:Pinecone ,embeddings:HuggingFaceEmbeddings) -> dict[str,Any]:
    try :
        loader=PyMuPDFLoader(file_path)
        docs=loader.load()

        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        text_chunks =text_splitter.split_documents(docs)
        
        existing_indexes = [index.name for index in pc.list_indexes()]
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )

        vector_store=PineconeVectorStore.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            index_name=index_name
        )
        return {
        "status": "success", 
        "message": "Documents uploaded successfully.",
        "chunks_created": len(text_chunks),
        "index_name": index_name
        },text_chunks

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during PDF ingestion: {str(e)}")
