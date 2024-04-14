import os
import gradio as gr
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from chain import qa_chain

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
VECTOR_DB_PATH = "./vectorstore/"
MODEL_API_KEY = os.getenv("OPENAI_API_KEY") or ""
    
def vector_search(provider, model_api_key, message):
    if provider == "Gemini":
        embedding = GoogleGenerativeAIEmbeddings(google_api_key=model_api_key, model="models/embedding-001", task_type="retrieval_query")
    else:
        embedding = OpenAIEmbeddings(openai_api_key=model_api_key, model="text-embedding-3-large", disallowed_special=())
    
    vector_db = Chroma(persist_directory=VECTOR_DB_PATH+provider, embedding_function=embedding)
    documents = vector_db.similarity_search(message)
    context = '\n'.join([doc.page_content for doc in documents])
    return context

def initialize_app(llm):
    config = RailsConfig.from_path("config")
    app = LLMRails(config=config, llm=llm)
    return app

def format_messages(message, relevant_chunks):
    messages = [{"role": "context", "content": {"relevant_chunks": relevant_chunks}}, {"role": "user", "content": message}]
    return messages

async def predict(message, _, model_api_key, provider, is_guardrails):
    if not model_api_key:
        return "OpenAI/Gemini API Key is required to run this demo, please enter your OpenAI API key in the settings and configs section!"

    if provider == "Gemini":
        llm = ChatGoogleGenerativeAI(google_api_key=model_api_key, model="gemini-pro")
    else:
        llm = ChatOpenAI(openai_api_key=model_api_key, model_name="gpt-3.5-turbo-16k")

    context = vector_search(provider, model_api_key, message)

    if not is_guardrails:
        return await qa_chain(llm, message, context)
    
    app = initialize_app(llm)
    response = await app.generate_async(messages=format_messages(message, context))
    return response["content"]


chat_textbox = gr.Textbox(placeholder="Hello, Ask any question related to AWS", container=False, scale=6)
examples = [["How to host my website in AWS?"], ["Tell me about AWS CodeStar?"]]

with gr.Blocks() as demo:
    bot = gr.Chatbot(height=600, render=False)
    gr.HTML("""<div style='height: 10px'></div>""")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                # AWS Chatbot | Guardrails
                Experiment on langchain with NeMo Guardrails.
                """
            )
        with gr.Column(scale=2):
            with gr.Group():
                with gr.Row():
                    guardrail = gr.Checkbox(label="Guardrails", info="Enables NeMo Guardrails",value=True, scale=0.5)
                    provider = gr.Dropdown(["OpenAI", "Gemini"], value="OpenAI", show_label=False, scale=1)
                    model_key = gr.Textbox(placeholder="Enter your OpenAI/Gemini API key", type="password", value=MODEL_API_KEY, show_label=False, scale=2)
                    

    gr.ChatInterface(
        predict, 
        chatbot=bot, 
        textbox=chat_textbox,  
        examples=examples, 
        theme="soft", 
        additional_inputs=[model_key, provider, guardrail]
    )

demo.launch()
