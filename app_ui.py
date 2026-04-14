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
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# HTML Meat Grinder
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

system_prompt = (
    "You are Nexus, a highly professional AI assistant. "
    "Use ONLY the following pieces of retrieved context to answer the user's question. "
    "If the answer is not in the context, say 'I cannot find the answer in the provided context'. "
    "Do not hallucinate.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 1. Startup is now clean and silent. It just waits for the user.
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content="👁️ **Nexus Controller Initialized.**\n\nDrop a URL in the chat OR click the paperclip icon to upload a PDF file to begin ingestion."
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    # ==========================================
    # PHASE 1: INGESTION MODE (No brain loaded yet)
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
                
                # Chainlit stores the uploaded file in a temporary path (file.path)
                loader = PyPDFLoader(file.path)
                docs = loader.load()
            else:
                await cl.Message(content="❌ Unsupported file type. Please upload a PDF.").send()
                return
        
        # Scenario B: User pasted a link
        elif message.content.startswith("http"):
            target_url = message.content.strip()
            msg.content = f"🌐 Detected URL: `{target_url}`. Scraping the web...\n"
            await msg.send()
            
            loader = RecursiveUrlLoader(url=target_url, max_depth=1, extractor=clean_html)
            docs = loader.load()
        
        # Scenario C: User just typed normal text
        else:
            await cl.Message(content="⚠️ You need to feed me data first! Paste a URL or upload a PDF.").send()
            return

        # --- The Math Engine (Same for both URLs and PDFs) ---
        if not docs:
            msg.content += "\n❌ Error: Could not extract any text. Try another file or link."
            await msg.update()
            return

        msg.content += f"\n✅ Extracted {len(docs)} pages. Slicing into chunks..."
        await msg.update()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        msg.content += "\n🧮 Calculating vector embeddings (Math time)..."
        await msg.update()
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(chunks, embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 3}) 

        llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        cl.user_session.set("rag_chain", rag_chain)

        msg.content += "\n\n🔥 **Ingestion Complete!** The Nexus Llama brain is fully loaded.\n\nWhat would you like to know about this data?"
        await msg.update()
        return

    # ==========================================
    # PHASE 2: CHAT MODE (Brain is active)
    # ==========================================
    res = cl.Message(content="Thinking...")
    await res.send()
    
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