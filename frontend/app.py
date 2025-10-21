import streamlit as st
import requests

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"

# Initialize session state
if 'document_uploaded' not in st.session_state:
    st.session_state.document_uploaded = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# App UI
st.title("🔍 NexusQuery RAG System")
st.write("Upload a PDF and ask questions about its content.")

# File upload section
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file and not st.session_state.document_uploaded:
    with st.spinner("Processing document..."):
        try:
            files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'application/pdf')}
            response = requests.post(f"{BACKEND_URL}/ingest/pdf", files=files)
            
            if response.status_code == 200:
                st.success("✅ Document processed successfully!")
                st.session_state.document_uploaded = True
            else:
                st.error(f"❌ Upload failed: {response.json().get('detail', 'Unknown error')}")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Please start the server.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Query section
if st.session_state.document_uploaded:
    st.divider()

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {question[:50]}..."):
                st.write(f"**Question:** {question}")
                st.write(f"**Answer:** {answer}")


    query = st.text_input("💭 Ask a question:", placeholder="What is this document about?")
    
    if st.button("Get Answer", type="primary"):
        if query:
            with st.spinner("Searching..."):
                try:
                    formatted_history = [[q, a] for q, a in st.session_state.chat_history]
                    payload = {
                        "query": query, 
                        "chat_history": st.session_state.chat_history
                    }
                    response = requests.post(f"{BACKEND_URL}/query", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        st.success("📝 Answer:")
                        st.write(answer)
                        
                        # Update chat history
                        st.session_state.chat_history = [(q, a) for q, a in result.get("chat_history", [])]

                        st.rerun()

                    else:
                        st.error(f"❌ Query failed: {response.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question.")

# Reset button
if st.session_state.document_uploaded:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Upload New Document"):
            st.session_state.document_uploaded = False
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()