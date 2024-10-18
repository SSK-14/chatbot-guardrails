from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from nemo.config import vector_search

OLLAMA_BASE_URL = "http://localhost:11434/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

def initialize_llm(model_api_key, provider, model):
    if provider == "gemini":
        return ChatGoogleGenerativeAI(google_api_key=model_api_key, model=model)
    elif provider == "openai":
        return ChatOpenAI(openai_api_key=model_api_key, model_name=model)
    elif provider == "groq":
        return ChatOpenAI(openai_api_key=model_api_key, openai_api_base=GROQ_BASE_URL, model_name=model)
    elif provider == "ollama":
        return ChatOpenAI(openai_api_key="", openai_api_base=OLLAMA_BASE_URL, model_name=model)
    else:
        return None

def prompt_template(question, context):
    return f"""You are an **GitDoc AI** Chatbot, a helpful assistant that assists users with their 
    NVIDIA's NeMo Guardrails related questions. 
    Use the following pieces of context to answer the user's question:
    {context}

    USER QUESTION: ```{question}```
    Answer in markdown:"""

def rag_chain(llm, message):
    context = vector_search(message)
    return llm.invoke(prompt_template(message, context)).content