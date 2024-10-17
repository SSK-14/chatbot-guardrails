import os
from nemoguardrails import LLMRails
from nemoguardrails.actions.actions import ActionResult
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()

COLLECTION_NAME = "nemo-docs" # Name of the collection
VECTOR_DB_PATH = "./qdrant" # Change this to your own path

qdrant_client = QdrantClient(path=VECTOR_DB_PATH)

# If using qdrant cloud, use the following code
# qdrant_client = QdrantClient(
#     os.getenv("QDRANT_URL"),
#     api_key=os.getenv("QDRANT_API_KEY"),
# )
  
def vector_search(query, limit=4):
    documents = qdrant_client.query(collection_name=COLLECTION_NAME, query_text=query, limit=limit)
    context = '\n\n'.join([f"PAGE_CONTENT: {doc.metadata['document']} SOURCE: {doc.metadata['source']}"  for doc in documents])
    return context

def prompt_template(question, context):
    return f"""You are an **GitDoc AI** Chatbot, a helpful assistant that assists users with their 
    **NVIDIA's NeMo Guardrails** related questions.
    CONTEXT INFORMATION is below.
    ---------------------
    {context}
    ---------------------

    RULES:
    1. Only Answer the USER QUESTION using the CONTEXT INFORMATION text above.
    2. Keep your answer grounded in the facts of the CONTEXT. 
    3. If you don't know the answer, just say that you don't know politely.
    4. Should not answer any out-of-context USER QUESTION.
    5. Add references only if needed in markdown format.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

def rag(context: dict, llm) -> ActionResult:
    context_updates = {}
    user_message = context.get("last_user_message")
    relevant_chunks = vector_search(user_message)
    context_updates["relevant_chunks"] = relevant_chunks
    prompt = prompt_template(user_message, relevant_chunks)
    answer = llm.invoke(prompt).content
    context_updates["_last_bot_prompt"] = prompt
    return ActionResult(return_value=answer, context_updates=context_updates)

def init(app: LLMRails):
    app.register_action(rag, "rag")