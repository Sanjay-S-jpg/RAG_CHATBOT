import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate 

# 1. Load the secret API key from the .env file so Groq lets us in
load_dotenv()

print("[*] Waking up the brain...")

# 2. We need the exact same math-translator to convert the user's question into math
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Load the saved database from your hard drive
# (allow_dangerous_deserialization is required for local pickle files, it's safe since you made it)
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Tell the DB to act as a search engine and return the top 3 most relevant chunks
retriever = vector_db.as_retriever(search_kwargs={"k": 3}) 

# 4. Fire up the Groq Engine (Using Llama 3 8B model - Crazy fast and 100% free)
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2)

# 5. The Prompt (The rules of the game for the AI)
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

# 6. Chain the search engine and the LLM together
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("[+] Brain connected. Ask a question about Claysys! (Type 'exit' to quit)")

# 7. The Chat Loop
while True:
    user_question = input("\nYou: ")
    if user_question.lower() == "exit":
        print("Catch ya later, brahh.")
        break
    
    # This runs the math search, grabs the text chunks, sends them to Groq, and prints the answer
    response = rag_chain.invoke({"input": user_question})
    print(f"\nHarlin-Bot: {response['answer']}")