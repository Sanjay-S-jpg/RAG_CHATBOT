# 🤖 Multi-Modal RAG Agent  
### Claysys AI Hackathon - Intelligence Division  

🌐 **Live Demo:** https://huggingface.co/spaces/sanjay33/Claysys-RAG-Agent

---

## 📖 Project Description  
The Multi-Modal RAG Agent is an enterprise-grade, zero-latency Retrieval-Augmented Generation (RAG) chatbot built for the Claysys AI Hackathon. It is designed to securely ingest, vectorize, and analyze both live web pages and PDF documents, providing users with hallucination-free answers backed by a verifiable audit trail.  

---

## 🧠 Solution Approach  
To ensure high accuracy and minimal latency, the architecture separates ingestion from generation. Documents and URLs are cleaned of HTML boilerplate, chunked into 1000-character overlapping segments to preserve context, and translated into mathematical vectors using local HuggingFace embeddings. When a user queries the bot, FAISS performs a real-time similarity search, retrieving only the top most relevant chunks. These chunks, along with the chat history, are fed to the Llama 3.1 model, forcing it to generate answers strictly grounded in the provided context.  

---

## ✨ Core Features 

- **Dynamic "Any-URL" Ingestion:**  
  Bypasses basic WAFs (like Cloudflare) using User-Agent spoofing to recursively scrape and clean live website data.  

- **Multi-Modal PDF Processing:**  
  Users can seamlessly upload documents directly through the UI for instant vectorization.  

- **Zero-Latency Generation:**  
  Hits the "minimal latency" requirement by leveraging Groq's specialized LPU hardware and the `llama-3.1-8b-instant` model.  

- **Hallucination-Free Audit Trail:**  
  Every AI response includes a "Verified Sources" dropdown, exposing the exact database chunks used to formulate the answer to guarantee accuracy.    

---

## 🛠️ Tech Stack Ecosystem  

- **Frontend:** Chainlit (Asynchronous, React-based Chat UI)  
- **LLM Engine:** Llama-3.1-8b-instant (via Groq API)  
- **Orchestration:** LangChain & LangChain Classic  
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local CPU/GPU processing)  
- **Vector Database:** FAISS (Facebook AI Similarity Search)  
- **Ingestion/Parsing:** `RecursiveUrlLoader`, `PyPDFLoader`, `BeautifulSoup4`  

---

## ⚙️ Setup and Usage Instructions  

### Prerequisites  

- Python 3.10+  
- A valid Groq API Key (https://console.groq.com/keys)

---

### 1. Clone the Repository  

```bash
git clone https://github.com/Sanjay-S-jpg/RAG_CHATBOT.git
cd RAG_chatbot
```

---

### 2. Initialize the Environment  

```bash
python -m venv rag_env
source rag_env/Scripts/activate  # Mac/Linux
rag_env\Scripts\activate         # Windows
pip install -r requirements.txt
```

---

### 3. Configure Credentials  

Create a `.env` file in the root directory and add your API key:  

```
GROQ_API_KEY=your_groq_api_key_here
```

---

### 4. Launch the Interface  

```bash
chainlit run app_ui.py -w
```

---

## 🚀 How to Use  

- Open the provided localhost link (`http://localhost:8000`) in your browser.  

- To analyze a website:  
  Paste a full URL (e.g., `https://claysys.com`) directly into the chat.  

- To analyze a document:  
  Click the paperclip icon in the chat bar and upload a PDF.  

- Wait for the **"Ingestion Complete"** confirmation.  

- Begin asking complex, context-specific questions about your data!  