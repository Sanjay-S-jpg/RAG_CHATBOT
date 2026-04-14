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

# --- NEW MEMORY IMPORTS ---
from langchain_classic.chains import create_history_aware_retriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

# HTML Meat Grinder
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

# ==========================================
# MEMORY PROMPTS
# ==========================================
# 1. Teaches the bot to rewrite your question using the history so it makes sense to the search engine
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 2. The main answering prompt, now with a placeholder for the history
system_prompt = (
    "You are Nexus, a highly professional AI assistant. "
    "Use ONLY the following pieces of retrieved context to answer the user's question. "
    "If the answer is not in the context, say 'I cannot find the answer in the provided context'. "
    "Do not hallucinate.\n\n"
    "Context:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])


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
                
                loader = PyPDFLoader(file.path)
                docs = loader.load()
            else:
                await cl.Message(content="❌ Unsupported file type. Please upload a PDF.").send()
                return
        
        # Scenario B: User pasted a link
        elif message.content.startswith("http"):
            target_url = message.content.strip()
            msg.content = f"🌐 Detected URL: `{target_url}`. Putting on the fake mustache and scraping the web...\n"
            await msg.send()
            
            # The Fake ID (User-Agent header)
            custom_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5"
            }
            
            # Wrap it in a try-except block so the UI doesn't crash if Cloudflare blocks us
            try:
                loader = RecursiveUrlLoader(
                    url=target_url, 
                    max_depth=1, 
                    extractor=clean_html,
                    headers=custom_headers # Passing the fake ID here!
                )
                docs = loader.load()
                
                if not docs:
                    await cl.Message(content="🛡️ **Blocked by Anti-Bot Security!**\nThis website has military-grade protection (like Cloudflare). Please try a different URL or upload a PDF instead.").send()
                    return

            except Exception as e:
                await cl.Message(content=f"⚠️ **Scraping Error:** The website rejected the connection or timed out. \n\n*Technical detail: {str(e)}*\n\nPlease try a different link or upload a PDF.").send()
                return
        
        # Scenario C: User just typed normal text
        else:
            await cl.Message(content="⚠️ You need to feed me data first! Paste a URL or upload a PDF.").send()
            return

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
        
        # --- WIRING THE MEMORY INTO THE CHAIN ---
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # Save the chain and initialize an EMPTY memory list
        cl.user_session.set("rag_chain", rag_chain)
        cl.user_session.set("chat_history", [])

        msg.content += "\n\n🔥 **Ingestion Complete!** The Nexus Llama brain is fully loaded.\n\nWhat would you like to know about this data?"
        await msg.update()
        return

    # ==========================================
    # PHASE 2: CHAT MODE (Brain is active)
    # ==========================================
    # Pull the history out of session memory
    chat_history = cl.user_session.get("chat_history")

    res = cl.Message(content="Thinking...")
    await res.send()
    
    # Pass both the input AND the history to the LLM
    response = rag_chain.invoke({
        "input": message.content,
        "chat_history": chat_history
    })
    
    answer = response["answer"]
    source_documents = response.get("context", []) 

    # --- THE 5-MESSAGE MEMORY CAP ---
    # Append the new Q&A to the history
    chat_history.extend([
        HumanMessage(content=message.content),
        AIMessage(content=answer)
    ])
    
    # Slicing: Keep only the last 10 messages (5 user questions + 5 AI answers)
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]
        
    # Save the updated history back to the session
    cl.user_session.set("chat_history", chat_history)

    # UI Sources Dropdown Logic
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