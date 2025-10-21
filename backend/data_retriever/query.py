from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from typing import Dict 
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone
from langchain.schema import HumanMessage, AIMessage

def get_answer_from_docs(query:str,chat_history:list,pc:Pinecone,embeddings:HuggingFaceEmbeddings,llm:ChatGroq,document_store:list[str]):
    try:
        pinecone_vector_store = PineconeVectorStore.from_existing_index(
        index_name="nexus-query-index",
        embedding=embeddings
        )

        pinecone_retriever=pinecone_vector_store.as_retriever()
        bm25_retriever=BM25Retriever.from_texts(document_store, k=5)
         
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, pinecone_retriever],
            weights=[0.5, 0.5]   
        )
        
        # Convert chat history to the format expected by ConversationalRetrievalChain
        formatted_chat_history = []
        for human_msg, ai_msg in chat_history:
            formatted_chat_history.extend([
                HumanMessage(content=human_msg),
                AIMessage(content=ai_msg)
            ])

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True ,output_key="answer" )

        # Add existing chat history to memory
        for message in formatted_chat_history:
            memory.chat_memory.add_message(message)

        retrieval_qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,  
         memory=memory,
            return_source_documents=True,
            verbose=True
        )

        # 4. Get the answer from the chain
        response = retrieval_qa_chain.invoke({"question": query })

        # Extract the new chat history from memory
        updated_chat_history = chat_history.copy()
        updated_chat_history.append((query, response.get('answer')))
        # messages = memory.chat_memory.messages
        # for i in range(0, len(messages), 2):
        #     if i + 1 < len(messages):
        #         human_content = messages[i].content
        #         ai_content = messages[i + 1].content
        #         updated_chat_history.append((human_content, ai_content))
                
        return {
        "answer": response.get('answer'),
        "source_documents": response.get('source_documents', []),
        "chat_history": updated_chat_history
        }
    
    except Exception as e:
       return {"error": f"An error occurred during retrieval: {str(e)}"}
