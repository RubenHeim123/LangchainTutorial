import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import time

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "odyssey.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exists. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    loader = TextLoader(file_path, autodetect_encoding=True)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    print("\n--- Creating embeddings ---")

    embeddings = FastEmbedEmbeddings(model_name='BAAI/bge-base-en-v1.5')
    print("\n--- Finished creating embeddings ---")

    print("\n--- Creating vector store ---")
    try:
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    except Exception as e:
        print("Rate limit exceeded. Retrying after 60 seconds...")
        time.sleep(60)  # Warte 60 Sekunden
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")