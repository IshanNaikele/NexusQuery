from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from typing import Dict 
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone

def get_answer_from_docs(query:str,chat_history:list,pc:Pinecone,embeddings:HuggingFaceEmbeddings,llm:ChatGroq,document_store:list[str]):
    try:
        pinecone_vector_store = PineconeVectorStore.from_existing_index(
        index_name="nexus-query-index",
        embedding=embeddings
        )

        pinecone_retriever=pinecone_vector_store.as_retriever()
        bm25_retriever=BM25Retriever.from_texts(document_store, k=5)
        bm25_retriever.k = 5
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, pinecone_retriever],
            weights=[0.5, 0.5]   
        )
         
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        retrieval_qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
         
        return_source_documents=True   
        )

        # 4. Get the answer from the chain
        response = retrieval_qa_chain.invoke({"question": query, "chat_history": chat_history})
        return {
        "answer": response.get('answer'),
        "source_documents": response.get('source_documents', []),
        "chat_history": chat_history
        }
    
    except Exception as e:
       return f"An error occurred during retrieval: {str(e)}"
