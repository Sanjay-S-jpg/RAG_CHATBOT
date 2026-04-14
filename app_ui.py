import os
import re
import chainlit as cl
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# HTML Cleaner
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

system_prompt = (
    "You are Nexus, a highly professional AI assistant. "
    "Use ONLY the following pieces of retrieved context to answer the user's question. "
    "If the answer is not in the context, say 'I cannot find the answer in the provided website data'. "
    "Do not hallucinate.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

@cl.on_chat_start
async def on_chat_start():
    # 1. Ask the judge for the URL right when they open the app
    url_request = await cl.AskUserMessage(
        content="👁️ **Nexus Controller Initialized.**\n\nPlease paste the target URL you want me to ingest and analyze:", 
        timeout=120
    ).send()

    if url_request:
        target_url = url_request["output"]
        
        # 2. Show a loading spinner so they know it's working
        msg = cl.Message(content=f"Scraping `{target_url}`... Please wait, this takes a moment.")
        await msg.send()

        try:
            # 3. Scrape the URL live!
            loader = RecursiveUrlLoader(url=target_url, max_depth=1, extractor=clean_html)
            docs = loader.load()
            
            if not docs:
                msg.content = "❌ Error: Could not extract any text from that URL. Try another one."
                await msg.update()
                return

            msg.content = f"✅ Extracted {len(docs)} pages. Slicing data into chunks..."
            await msg.update()

            # 4. Chunk it
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)

            # 5. Embed and Vectorize (In RAM, so it's fresh for every new link)
            msg.content = "🧮 Calculating vector embeddings..."
            await msg.update()
            
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vector_db = FAISS.from_documents(chunks, embeddings)
            retriever = vector_db.as_retriever(search_kwargs={"k": 3}) 

            # 6. Connect the Brain
            llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)

            # Save to session memory
            cl.user_session.set("rag_chain", rag_chain)

            msg.content = f"🔥 **Ingestion Complete!** The Llama 3.1 brain has memorized `{target_url}`.\n\nWhat would you like to know about it?"
            await msg.update()

        except Exception as e:
            msg.content = f"❌ An error occurred: {str(e)}"
            await msg.update()

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    if not rag_chain:
        await cl.Message(content="Please provide a valid URL first!").send()
        return

    res = cl.Message(content="Thinking...")
    await res.send()
    
    # 1. Ask the Llama the question (this returns a dictionary with the answer AND the source chunks)
    response = rag_chain.invoke({"input": message.content})
    
    answer = response["answer"]
    source_documents = response["context"] # These are the raw chunks from the website

    # 2. Build the UI Dropdown Elements for the Sources
    text_elements = []
    if source_documents:
        for source_idx, doc in enumerate(source_documents):
            source_name = f"Source Chunk {source_idx + 1}"
            # We added display="side" here so it hides in a drawer until clicked!
            text_elements.append(
                cl.Text(content=doc.page_content, name=source_name, display="side")
            )
        
        # Add clickable reference names to the bottom of the bot's message
        source_names = [text_el.name for text_el in text_elements]
        answer += f"\n\n**🔍 Verified Sources:** {', '.join(source_names)}"
    else:
        answer += "\n\n*No sources found in the database.*"

    # 3. Update the message on the screen with the text and the dropdown elements
    res.content = answer
    res.elements = text_elements
    await res.update()