# 🧠 Multi-Agent LangGraph System

This project demonstrates a multi-agent system using [LangGraph](https://github.com/langchain-ai/langgraph), a framework designed for building stateful, multi-agent applications powered by LLMs. The system showcases how multiple specialized agents can collaborate in an E-commerce context to solve user queries effectively.

## 🚀 Project Overview

This notebook implements a LangGraph-based multi-agent architecture that includes:

- **Memory Node**: Extracts and maintains conversation history and key memory chunks.
- **QA Agent**: Answers factual questions based on user input and relevant memory.
- **Summary Node**: Summarizes the entire interaction for logging or user feedback.

### 🛒 Use Case
The agents are tested within an **E-commerce** scenario (e.g., smartwatch sales), making it applicable to customer support, shopping assistants, and order tracking systems.

---

## 🧩 Components

| Component             | Description |
|-----------------------|-------------|
| `Memory Node`         | Stores and retrieves conversation memory. |
| `QA Agent`            | Answers general and factual questions. |
| `Summary Node`        | Outputs a final summary of all agent contributions. |

---

## 🛠️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Ibrahimnasser2/Multi-Agent-LangGraph-System-E-commerce.git
cd Multi-Agent-LangGraph
