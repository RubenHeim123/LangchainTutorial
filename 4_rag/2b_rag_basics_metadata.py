import os

from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "db_with_metadata")

embeddings = FastEmbedEmbeddings()

db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

query = "How did Juliet die?"

retriever = db.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={'k':3, 'score_threshold':0.1}
)
relevant_docs = retriever.invoke(query)

print('\n--- Relevant Documents ---')
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document{i}: \n{doc.page_content}\n")
    print(f"Source: {doc.metadata.get('source','Unknown')}\n")