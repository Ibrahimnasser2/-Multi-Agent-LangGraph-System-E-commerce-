# 🧠 CogniGraph - Intelligent Document Analysis System

This project demonstrates a multi-agent document analysis system using [LangGraph](https://github.com/langchain-ai/langgraph) with automatic conversation summarization and knowledge extraction capabilities.

## 🌟 Key Features

- **Document Intelligence**: Process PDFs and web content with advanced RAG capabilities
- **Conversation Memory**: Maintains context across interactions
- **Auto-Summarization**: Generates concise summaries every 5 messages
- **Knowledge Extraction**: Identifies and highlights key information
- **Visual Analytics**: Beautiful dashboard with conversation insights

## 🚀 Project Overview

This implementation features a LangGraph-based architecture with:

- **Memory Node**: Extracts and maintains user information and conversation history
- **QA Agent**: Answers questions based on document content using FAISS vector search
- **Summary Node**: Automatically generates bullet-point summaries of conversations
- **Visual Dashboard**: Streamlit interface with dedicated knowledge panel

### 📊 Use Cases
- **Document Q&A Systems**
- **Research Assistant Tools**
- **Knowledge Management Platforms**
- **Customer Support Automation**

---

## 🧩 System Architecture

| Component             | Description | Technology |
|-----------------------|-------------|------------|
| Memory Node         | Stores user details and conversation context | LangGraph |
| QA Agent            | Answers document-based questions | HuggingFace (flan-t5-large) |
| Summary Node        | Generates conversation summaries | Transformers |
| Vector Store        | Document indexing and retrieval | FAISS |
| Frontend           | Interactive dashboard | Streamlit |

![System Diagram](https://via.placeholder.com/800x400?text=CogniGraph+System+Architecture)

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/CogniGraph.git
cd CogniGraph
