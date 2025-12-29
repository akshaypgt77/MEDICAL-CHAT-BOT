from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter



#extract text from pdf file
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


#Filter to minimal
from typing import List 
from langchain.schema import Document

def filter_to_minimal_docs(docs: List[Document]) -> List[Document] :
    """ 
    Given a list of Document objects, return a new list of Document objects 
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get ("source")
        minimal_docs.append(
            Document (
                page_content=doc.page_content, 
                metadata={"source": src}
            )
        )
    return minimal_docs

#CHUNKING splitting into smaller chunks

def text_split(minimal_docs) :
    text_splitter = RecursiveCharacterTextSplitter(
         chunk_size=500, 
         chunk_overlap=20,
         length_function=len
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


#Downloading the embedding model
from langchain_community.embeddings import HuggingFaceEmbeddings

def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embedding

embedding = download_embeddings()


