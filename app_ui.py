import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load API keys
load_dotenv()

# The Prompt (Rules of the game)
system_prompt = (
    "You are a highly professional AI assistant for Claysys. "
    "Use ONLY the following pieces of retrieved context to answer the user's question. "
    "If the answer is not in the context, say 'I don't know based on the provided website data'. "
    "Do not hallucinate or make things up.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# This runs ONE TIME when the user opens the webpage
@cl.on_chat_start
async def on_chat_start():
    # Show a loading message in the UI
    msg = cl.Message(content="Waking up the Llama 3.1 brain... Give me a sec, brahh.")
    await msg.send()

    # 1. Load the math-translator
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2. Load the saved Vector DB from Phase 3
    vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Grab the top 3 most relevant chunks
    retriever = vector_db.as_retriever(search_kwargs={"k": 3}) 

    # 3. Fire up Groq Engine
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)

    # 4. Chain them together
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 5. Save the brain in the user's browser session memory
    cl.user_session.set("rag_chain", rag_chain)

    # Update the loading message
    msg.content = "Brain connected! 🧠 What do you want to know about Claysys?"
    await msg.update()

# This runs EVERY TIME the user types a message and hits enter
@cl.on_message
async def on_message(message: cl.Message):
    # 1. Grab the brain from memory
    rag_chain = cl.user_session.get("rag_chain")
    
    # 2. Send an empty message box to the UI (so we can fill it in)
    res = cl.Message(content="Thinking...")
    await res.send()
    
    # 3. Ask the Llama the question
    response = rag_chain.invoke({"input": message.content})
    
    # 4. Update the empty message box with the real answer
    res.content = response["answer"]
    await res.update()