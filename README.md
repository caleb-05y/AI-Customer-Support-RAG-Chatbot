# 🤖 AI Business & Support Assistant (RAG Chatbot)

A production-style **Retrieval-Augmented Generation (RAG)** chatbot that answers business intelligence and customer support questions using a custom knowledge base — with **source-grounded responses**.

---

## 🚀 Overview

This project demonstrates how to build a real-world AI system that:
- Retrieves relevant information from documents
- Uses LLMs to generate accurate answers
- Avoids hallucinations by grounding responses in data

The chatbot can answer questions about:
- 📊 Business analytics (e.g., customer churn)
- 💼 KPIs and metrics
- 🧾 Customer support knowledge

---

## 🧠 How It Works (RAG Pipeline)

1. **Document Ingestion**
   - Load `.txt` / `.pdf` files
   - Split into chunks
   - Generate embeddings using OpenAI

2. **Vector Search (FAISS)**
   - Store embeddings in a FAISS vector database
   - Retrieve top-k relevant chunks per query

3. **LLM Response Generation**
   - Use `gpt-4o-mini`
   - Strict prompt to ensure answers come only from context

4. **Streamlit UI**
   - Interactive chat interface
   - Displays answers + source documents

---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **OpenAI API**
- **FAISS (Vector Database)**
- **Streamlit**
- **Pandas / NumPy**

---

## 📁 Project Structure
# 🤖 AI Business & Support Assistant (RAG Chatbot)

A production-style **Retrieval-Augmented Generation (RAG)** chatbot that answers business intelligence and customer support questions using a custom knowledge base — with **source-grounded responses**.

---

## 🚀 Overview

This project demonstrates how to build a real-world AI system that:
- Retrieves relevant information from documents
- Uses LLMs to generate accurate answers
- Avoids hallucinations by grounding responses in data

The chatbot can answer questions about:
- 📊 Business analytics (e.g., customer churn)
- 💼 KPIs and metrics
- 🧾 Customer support knowledge

---

## 🧠 How It Works (RAG Pipeline)

1. **Document Ingestion**
   - Load `.txt` / `.pdf` files
   - Split into chunks
   - Generate embeddings using OpenAI

2. **Vector Search (FAISS)**
   - Store embeddings in a FAISS vector database
   - Retrieve top-k relevant chunks per query

3. **LLM Response Generation**
   - Use `gpt-4o-mini`
   - Strict prompt to ensure answers come only from context

4. **Streamlit UI**
   - Interactive chat interface
   - Displays answers + source documents

---

## 🛠️ Tech Stack

- **Python**
- **LangChain**
- **OpenAI API**
- **FAISS (Vector Database)**
- **Streamlit**
- **Pandas / NumPy**

---

## 📁 Project Structure

rag-chatbot/
├── app.py # Streamlit UI
├── ingest.py # Builds vector database
├── rag_pipeline.py # RAG logic
├── data/ # Knowledge base files
├── vectorstore/ # FAISS index (auto-generated)
├── requirements.txt
└── README.md

---

## ⚙️ Setup Instructions

### 1. Clone repo
```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
Create virtual environment
python3 -m venv venv
source venv/bin/activate
Install dependencies
pip install -r requirements.txt
Add OpenAI API key
OPENAI_API_KEY=your_key_here
Build vector database
python ingest.py
Run the app
streamlit run app.py
```
## 💬 Example Questions
What is customer churn?
How do you reduce churn?
What are important KPIs in business analytics?
What payment methods are accepted?
## ✨ Key Features
🔍 Semantic search over documents
🧠 Context-aware answers using RAG
📄 Source citations for every response
💬 Multi-turn conversation support
⚡ Fast local vector search with FAISS
## 📈 Improvements Made
Tuned retrieval (top-k) for better relevance
Removed noisy documents to improve answer quality
Added query preprocessing for cleaner retrieval
Implemented source formatting and deduplication
## 🚀 Future Improvements
Deploy app (Streamlit Cloud / AWS)
Add file upload for dynamic knowledge bases
Improve UI with chat bubbles + better UX
Add domain-based routing (business vs support)
Track answer confidence scores
## 🧑‍💻 Author
Caleb Youhanna
