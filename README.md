# 📄 PDF Academic Question Answering (RAG)

A PDF-based Question Answering system built using Retrieval-Augmented Generation (RAG). Users can chat with academic PDF documents and receive accurate, source-grounded answers.

## Features
- Chat with PDF documents  
- Semantic search using embeddings  
- Conversational memory for follow-up questions  
- Answers strictly from PDF content  
- Source document and page number citations  
- Streamlit chat interface  

## Tech Stack
Python, Streamlit, LangChain, LangGraph, ChromaDB, HuggingFace Embeddings, Groq LLaMA-3.1

## How to Run
```bash
git clone <repo-url>
cd <repo-name>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

Create .env:

GROQ_API_KEY=your_api_key

Add PDFs to data/, then:

python ingest.py
streamlit run app.py


Open: http://localhost:8501

Notes

If the answer is not in the PDF, the system responds with "I don't know"

Designed to prevent hallucinations

License

Educational use only.
