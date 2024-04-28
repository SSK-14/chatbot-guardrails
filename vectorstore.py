import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()

PATH_TO_KNOWLEDGE_BASE = "knowledge_base" # Path where the PDFs are stored
COLLECTION_NAME = "aws_faq" # Name of the collection
VECTOR_DB_PATH = "./qdrant" # Change this to your own path

qdrant_client = QdrantClient(path=VECTOR_DB_PATH)

# If using qdrant cloud, use the following code
# qdrant_client = QdrantClient(
#     os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY"),
# )

def ingest_embeddings():
    documents = []
    for file in os.listdir(PATH_TO_KNOWLEDGE_BASE):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(PATH_TO_KNOWLEDGE_BASE, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=300, chunk_overlap=50)
    chunked_documents = text_splitter.split_documents(documents)
    chunks, metadata, ids = zip(*[(chunk.page_content, chunk.metadata, i+1) for i, chunk in enumerate(chunked_documents)])
    try:
        qdrant_client.add(
            collection_name=COLLECTION_NAME,
            documents=chunks,
            metadata=metadata,
            ids=ids
        )

        print("Collection created and persisted")
    except Exception as error:
        print(f"Error: {error}")

if __name__ == "__main__":
    ingest_embeddings()