import os
import re
import chainlit as cl
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import RecursiveUrlLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- BACK TO BASICS: PURE RAG IMPORTS ---
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# HTML Meat Grinder
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

# ==========================================
# THE PURE RAG PROMPT (No Memory BS)
# ==========================================
system_prompt = (
    "You are Nexus, a highly professional AI assistant. "
    "Use ONLY the following pieces of retrieved context to answer the user's question. "
    "If the answer is not in the context, say 'I cannot find the answer in the provided data'. "
    "Do not hallucinate.\n\n"
    "Context:\n{context}"
)

# No chat_history placeholders. Just pure input and output.
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="👁️ **RAG AGENT Initialized.**\n\nDrop a URL in the chat OR click the paperclip icon to upload a PDF file to begin ingestion."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    # ==========================================
    # PHASE 1: INGESTION MODE
    # ==========================================
    if not rag_chain:
        msg = cl.Message(content="")
        docs = []
        
        # Scenario A: User uploaded a file
        if message.elements:
            file = message.elements[0]
            if file.name.endswith(".pdf"):
                msg.content = f"📄 Detected PDF: `{file.name}`. Extracting text...\n"
                await msg.send()
                
                loader = PyPDFLoader(file.path)
                docs = loader.load()
            else:
                await cl.Message(content="❌ Unsupported file type. Please upload a PDF.").send()
                return
        
        # Scenario B: User pasted a link
        elif message.content.startswith("http"):
            target_url = message.content.strip()
            msg.content = f"🌐 Detected URL: `{target_url}`. Analyzing target security...\n"
            await msg.send()
            
            # THE SMART ROUTER (Kept this so Cloudflare doesn't nuke us)
            if "wikipedia.org" in target_url:
                req_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
                msg.content += "🕵️‍♂️ Strict robot policy detected. Equipping User-Agent mask...\n"
            else:
                req_headers = None
                msg.content += "🚀 Standard target detected. Proceeding with native ingestion...\n"
            
            await msg.update()

            try:
                loader = RecursiveUrlLoader(
                    url=target_url, 
                    max_depth=1, 
                    extractor=clean_html,
                    headers=req_headers
                )
                docs = loader.load()
                
                if docs and "Enable JavaScript and cookies to continue" in docs[0].page_content:
                    await cl.Message(content="🧱 **Cloudflare Wall Hit!**\nThis website requires a real browser engine to execute JavaScript. Please upload a PDF or try a different URL.").send()
                    return
                
                if not docs:
                    await cl.Message(content="🛡️ **Blocked by Security!**\nCould not extract text. Please try a different URL or upload a PDF instead.").send()
                    return

            except Exception as e:
                await cl.Message(content=f"⚠️ **Scraping Error:** The website rejected the connection. \n\n*Technical detail: {str(e)}*\n\nPlease try a different link or upload a PDF.").send()
                return
        
        else:
            await cl.Message(content="⚠️ You need to feed me data first! Paste a URL or upload a PDF.").send()
            return

        if not docs:
            msg.content += "\n❌ Error: Could not extract any text. Try another file or link."
            await msg.update()
            return

        msg.content += f"\n✅ Extracted {len(docs)} pages. Slicing into chunks..."
        await msg.update()
        
        # THE SNIPER FIX: Dense chunks so addresses don't get lost
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(docs)

        msg.content += "\n🧮 Calculating vector embeddings (Math time)..."
        await msg.update()
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        
        # THE SHOTGUN FIX: Pulling 10 chunks to guarantee we catch the right data
        retriever = vector_db.as_retriever(search_kwargs={"k": 10}) 

        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)
        
        # PURE CHAIN: Straight from the database to the Llama. Zero hallucination.
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        cl.user_session.set("rag_chain", rag_chain)

        msg.content += "\n\n🔥 **Ingestion Complete!** The RAGAgent Llama brain is fully loaded.\n\nWhat would you like to know about this data?"
        await msg.update()
        return

    # ==========================================
    # PHASE 2: CHAT MODE (Pure, Stateless RAG)
    # ==========================================
    res = cl.Message(content="Thinking...")
    await res.send()
    
    # We just feed it the raw input. No memory dict required.
    response = rag_chain.invoke({"input": message.content})
    
    answer = response["answer"]
    source_documents = response.get("context", []) 

    text_elements = []
    if source_documents:
        for source_idx, doc in enumerate(source_documents):
            source_name = f"Source Chunk {source_idx + 1}"
            text_elements.append(
                cl.Text(content=doc.page_content, name=source_name, display="side")
            )
        
        source_names = [text_el.name for text_el in text_elements]
        answer += f"\n\n**🔍 Verified Sources:** {', '.join(source_names)}"
    else:
        answer += "\n\n*No sources found in the database.*"

    res.content = answer
    res.elements = text_elements
    await res.update()