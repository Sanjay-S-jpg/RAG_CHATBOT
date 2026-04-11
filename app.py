from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
import re

# 1. Clean the HTML trash
def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

# 2. Scrape the website
def get_website_data(url: str):
    print(f"[*] Scraping {url}...")
    loader = RecursiveUrlLoader(url=url, max_depth=1, extractor=clean_html)
    docs = loader.load()
    print(f"[+] Scraped {len(docs)} pages.")
    return docs

# 3. Chunking and Embeddings (The Free Math part)
def create_vector_db(docs):
    print("[*] Slicing text into chunks...")
    # We slice the text into 1000-character chunks so the LLM can digest it
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    print("[*] Downloading free HuggingFace embedding model (one-time setup)...")
    # This uses your CPU/GPU locally for free
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("[*] Creating FAISS Vector Database...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    print("[+] Vector Database locked and loaded!")
    
    vector_db.save_local("faiss_index")
    return vector_db

if __name__ == "__main__":
    target_url = "https://claysys.com"
    raw_docs = get_website_data(target_url)
    
    # Pass the scraped data into our vector database creator
    db = create_vector_db(raw_docs)