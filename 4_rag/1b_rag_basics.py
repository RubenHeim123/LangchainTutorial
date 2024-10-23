import os

from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

embeddings = FastEmbedEmbeddings(model_name='BAAI/bge-base-en-v1.5')

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "Who is Odysseus wife?"

retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k':3, 'score_threshold':0.4}
)
relevant_docs = retriever.invoke(query)

print('\n--- Relevant Documents ---')
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document{i}: \n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source','Unknown')}\n")