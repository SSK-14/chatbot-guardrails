import os
import gradio as gr
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from chain import qa_chain
from vectorstore import qdrant_client

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_API_KEY = os.getenv("OPENAI_API_KEY") or ""
    
def vector_search(message):
    documents = qdrant_client.query(collection_name="aws_faq", query_text=message, limit=4)
    print(documents)
    context = '\n'.join([doc.metadata["document"] for doc in documents])
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

    context = vector_search(message)

    if not is_guardrails:
        return await qa_chain(llm, message, context)
    
    app = initialize_app(llm)
    response = await app.generate_async(messages=format_messages(message, context))
    return response["content"]


chat_textbox = gr.Textbox(placeholder="Hello, Ask any question related to AWS EC2 or S3", container=False, scale=6)
examples = [["How reliable is Amazon S3?"], ["How do I get started with EC2 Capacity Blocks?"]]

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
