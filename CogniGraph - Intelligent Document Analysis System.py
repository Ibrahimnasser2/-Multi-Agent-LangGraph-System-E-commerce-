# --- backend.py ---
import os
from langgraph.graph import Graph
from typing import Dict, Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RAGManagerFAISS:
    def __init__(self, embedding_model=None, faiss_index_path="faiss_index", llm_model_name="google/flan-t5-large"):
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.faiss_index_path = faiss_index_path
        self.faiss_store = None
        self._load_index()

        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)
        self.qa_pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)

    def load_documents(self, file_paths=[], urls=[]):
        documents = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        for url in urls:
            loader = WebBaseLoader(url)
            documents.extend(loader.load())
        return documents

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(documents)

    def index_documents(self, documents):
        chunks = self.split_documents(documents)
        if self.faiss_store is None:
            self.faiss_store = FAISS.from_documents(chunks, self.embedding_model)
        else:
            self.faiss_store.add_documents(chunks)
        self._save_index()

    def query(self, query_text, top_k=1):
        if not self.faiss_store:
            raise ValueError("No FAISS index found. Please index documents first!")
        results = self.faiss_store.similarity_search(query_text, k=top_k)
        return [result.page_content for result in results]

    def generate_answer(self, question, top_k=3):
        if not self.faiss_store:
            return "No indexed documents to generate answers from."
        retrieved_docs = self.faiss_store.similarity_search(question, k=top_k)
        if not retrieved_docs:
            return "No relevant information found."
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        prompt = f"Question: {question}\nContext: {context}"
        output = self.qa_pipeline(prompt, max_length=256, temperature=0.2)
        return output[0]['generated_text']

    def _save_index(self):
        if self.faiss_store:
            os.makedirs(self.faiss_index_path, exist_ok=True)
            self.faiss_store.save_local(self.faiss_index_path)

    def _load_index(self):
        if os.path.exists(self.faiss_index_path):
            try:
                self.faiss_store = FAISS.load_local(self.faiss_index_path, self.embedding_model, allow_dangerous_deserialization=True)
            except Exception:
                self.faiss_store = None

user_memory = {}
rag_manager = RAGManagerFAISS()

def memory_extraction_node(state: Dict[str, Any]) -> Dict[str, Any]:
    messages = state.get("messages", [])
    if not messages:
        return state
    message = messages[-1]
    user_message = message.get("content", "") if isinstance(message, dict) else str(message)
    if "my name is" in user_message.lower():
        user_name = user_message.split("my name is")[-1].strip().split()[0]
        user_memory['name'] = user_name
    state["memory"] = user_memory
    return state

def rag_node(state: Dict[str, Any]) -> Dict[str, Any]:
    messages = state.get("messages", [])
    if not messages:
        return state
    last_message = messages[-1]
    user_question = last_message.get("content", "") if isinstance(last_message, dict) else str(last_message)
    answer = rag_manager.generate_answer(user_question, top_k=3)
    state["messages"].append({"role": "assistant", "content": answer})
    return state

def summary_node(state: Dict[str, Any]) -> Dict[str, Any]:
    messages = state.get("messages", [])
    if len(messages) >= 5:
        context = "\n".join(
            msg.get("content", "") if isinstance(msg, dict) else str(msg)
            for msg in messages
        )
        prompt = f"Extract key information and create a concise summary of this conversation:\n{context}"
        summary = rag_manager.qa_pipeline(prompt, max_length=200)[0]['generated_text']
        state["messages"] = [{"role": "assistant", "content": "I've updated the conversation summary. See the summary panel for details."}]
        state["summary"] = summary
        state["last_summary_time"] = time.time()
    return state

graph = Graph()
graph.add_node("MemoryExtraction", memory_extraction_node)
graph.add_node("RAGQA", rag_node)
graph.add_node("Summary", summary_node)
graph.add_edge("MemoryExtraction", "RAGQA")
graph.add_edge("RAGQA", "Summary")
graph.set_entry_point("MemoryExtraction")
graph.set_finish_point("Summary")
graph.compile()

# --- app.py ---
# --- app.py --- (corrected version)
import streamlit as st
from streamlit_chat import message
import time
import random

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="🧠 CogniSearch - Knowledge Extraction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the summary card
st.markdown("""
<style>
    .summary-container {
        position: relative;
        margin: 20px 0;
        border-radius: 12px;
        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        padding: 0;
        overflow: hidden;
        border-left: 5px solid #2c3e50;
    }
    .summary-header {
        background-color: #2c3e50;
        color: white;
        padding: 12px 20px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .summary-content {
        padding: 20px;
        background-color: white;
        border-radius: 0 0 12px 12px;
    }
    .summary-badge {
        position: absolute;
        top: -10px;
        right: 20px;
        background-color: #e74c3c;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    div[data-testid="stChatMessageUser"] {
        background-color: #eaf2f8 !important;
        border-left: 4px solid #3498db !important;
    }
    div[data-testid="stChatMessageAssistant"] {
        background-color: #e8f8f5 !important;
        border-left: 4px solid #1abc9c !important;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50; margin-bottom: 10px;'>
        🧠 CogniSearch - Knowledge Extraction System
    </h1>
    <p style='text-align: center; color: #7f8c8d; margin-bottom: 30px;'>
        Advanced document analysis with automatic conversation summarization
    </p>
""", unsafe_allow_html=True)

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.conversation_summaries = []
    st.session_state.last_summary_time = None

if "graph_state" not in st.session_state:
    st.session_state.graph_state = {"messages": []}

if "indexed" not in st.session_state:
    st.session_state.indexed = False

# Sidebar
with st.sidebar:
    st.header("📂 Data Sources")
    uploaded_file = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    url_input = st.text_input("Or enter website URLs (comma separated)")
    
    if st.button("Process Documents", type="primary"):
        if uploaded_file or url_input:
            with st.spinner("Indexing documents..."):
                docs = []
                if uploaded_file:
                    for file in uploaded_file:
                        with open(f"./{file.name}", "wb") as f:
                            f.write(file.getbuffer())
                        docs.extend(rag_manager.load_documents(file_paths=[f"./{file.name}"]))
                if url_input:
                    urls = [url.strip() for url in url_input.split(",") if url.strip()]
                    docs.extend(rag_manager.load_documents(urls=urls))
                
                if docs:
                    rag_manager.index_documents(docs)
                    st.session_state.indexed = True
                    st.success("Documents successfully indexed!")
                    st.balloons()
        else:
            st.warning("Please upload files or enter URLs")

# Main Content
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Conversation")
    
    # Chat input
    user_input = st.chat_input("Ask about your documents...")
    
    if user_input:
        if not st.session_state.indexed:
            st.warning("Please index documents first", icon="⚠️")
        else:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            st.session_state.graph_state["messages"].append({"role": "user", "content": user_input})
            
            with st.spinner("Analyzing..."):
                final_state = graph.compile().invoke(st.session_state.graph_state)
                st.session_state.graph_state = final_state
                
                assistant_reply = final_state["messages"][-1]["content"]
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_reply})
    
    # Display chat
    for msg in st.session_state.chat_history:
        message(msg["content"], is_user=(msg["role"] == "user"), key=f"{msg['content']}_{msg['role']}")

with col2:
    st.subheader("Knowledge Panel")
    
    # Summary Card
    if "summary" in st.session_state.graph_state:
        summary = st.session_state.graph_state["summary"]
        st.session_state.conversation_summaries.append(summary)
        
        st.markdown(f"""
        <div class="summary-container">
            <div class="summary-header">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                </svg>
                EXTRACTED KNOWLEDGE
            </div>
            <div class="summary-content">
                {summary}
            </div>
            <div class="summary-badge">
                NEW
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Summary history
        with st.expander("View History", expanded=False):
            if len(st.session_state.conversation_summaries) > 1:
                for i, prev_summary in enumerate(reversed(st.session_state.conversation_summaries[:-1])):
                    st.markdown(f"**Summary {len(st.session_state.conversation_summaries)-i-1}**")
                    st.caption(prev_summary)
                    if i < len(st.session_state.conversation_summaries)-2:
                        st.divider()
            else:
                st.caption("No previous summaries yet")
    
    else:
        st.info("Conversation summary will appear here after 5 messages", icon="ℹ️")

# --- backend.py --- (keep exactly the same as before)