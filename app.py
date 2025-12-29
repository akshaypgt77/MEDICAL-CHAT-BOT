from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

from src.helper import download_embeddings
from src.prompt import system_prompt

from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# -------------------- FLASK INIT --------------------
app = Flask(__name__)

# -------------------- LOAD ENV --------------------
load_dotenv()

# -------------------- EMBEDDINGS --------------------
embeddings = download_embeddings()

# -------------------- PINECONE --------------------
index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_kwargs={"k": 3})

# -------------------- LLM --------------------
chatModel = ChatOpenAI(model="gpt-4o-mini")

# -------------------- PROMPT --------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# -------------------- RAG CHAIN --------------------
qa_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# -------------------- ROUTES --------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    user_input = data.get("question")

    response = rag_chain.invoke({"input": user_input})
    return jsonify({"answer": response["answer"]})

# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
