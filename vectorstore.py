import os, zipfile, requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

COLLECTION_NAME = "nemo-docs" # Name of the collection
VECTOR_DB_PATH = "./qdrant" # Change this to your own path
GITHUB_URL = "https://github.com/NVIDIA/NeMo-Guardrails"
BRANCH = "develop"

# qdrant_client = QdrantClient(path=VECTOR_DB_PATH)

# If using qdrant cloud, use the following code
qdrant_client = QdrantClient(
    os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)

def load_github_docs(document_url, branch='master'):
    filename = os.path.basename(document_url.rstrip('/').strip())
    unzip_path = 'docs' + '/' + filename + '-' + branch
    document_url = document_url + "/archive/refs/heads/" + branch + ".zip"
    temp_dir = os.path.join(os.getcwd(), 'docs')
    response = requests.get(document_url)
    if response.status_code == 200:
        zip_data = BytesIO(response.content)
        with zipfile.ZipFile(zip_data, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        return unzip_path
    else:
        return None

def ingest_embeddings(path):
    metadatas = []
    text = []
    for root, _, files in os.walk(path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(file_path, path)
            relative_path = GITHUB_URL + "/blob/" + BRANCH + '/' + relative_path
            try:
                if file_name.endswith(".md"):
                    with open(file_path, "r", encoding="utf-8") as file:
                        text.append(file.read())
                        metadatas.append({"source": relative_path})
            except Exception as error:
                print(f"Error: {error}")            
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=700, chunk_overlap=100)
    chunked_documents = text_splitter.create_documents(text, metadatas=metadatas)
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
        
def vector_search(query, limit=4):
    documents = qdrant_client.query(collection_name=COLLECTION_NAME, query_text=query, limit=limit)
    context = '\n'.join([doc.metadata["document"] for doc in documents])
    return context

if __name__ == "__main__":
    file_path = load_github_docs(GITHUB_URL, BRANCH)
    ingest_embeddings(file_path)