import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from dotenv import load_dotenv
load_dotenv()

PATH_TO_KNOWLEDGE_BASE = "knowledge_base" # Path where the PDFs are stored
COLLECTION_NAME = "aws_docs" # Name of the collection
VECTOR_DB_PATH = "./temp/vectorstore" # Change this to your own path

embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), disallowed_special=())

def create_collection():
    documents = []
    for file in os.listdir(PATH_TO_KNOWLEDGE_BASE):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(PATH_TO_KNOWLEDGE_BASE, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    try:
        client = chromadb.Client()
        client.create_collection(COLLECTION_NAME)
        vector_db = Chroma.from_documents(
            documents=chunked_documents,
            embedding=embedding,
            persist_directory=VECTOR_DB_PATH,
        )
        vector_db.persist()
        print("Collection created and persisted")
    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    create_collection()