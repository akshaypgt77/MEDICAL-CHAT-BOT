print("STORE_INDEX.PY IS RUNNING")
exit()

import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.helper import (
    load_pdf_files,
    filter_to_minimal_docs,
    text_split,
    download_embeddings
)

print("Script started")

# ------------------ API KEY ------------------
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
print("API KEY FOUND:", bool(PINECONE_API_KEY))

if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY NOT SET")

pc = Pinecone(api_key=PINECONE_API_KEY)

# ------------------ INDEX ------------------
index_name = "medical-chatbot"

existing_indexes = [i.name for i in pc.list_indexes()]
print("Existing indexes:", existing_indexes)

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Index CREATED")
else:
    print("Index already exists")

# ------------------ LOAD DATA ------------------
docs = load_pdf_files("data")
docs = filter_to_minimal_docs(docs)
chunks = text_split(docs)
print(f"📄 Chunks created: {len(chunks)}")

# ------------------ EMBEDDINGS ------------------
embeddings = download_embeddings()
print("Embeddings loaded")

# ------------------ UPSERT ------------------
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

print("VECTORS STORED SUCCESSFULLY")
