import os
import gradio as gr
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

VECTOR_DB_PATH = "./temp/vectorstore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""
    
def vector_search(openai_api_key, message):
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, disallowed_special=())
    vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding)
    documents = vector_db.similarity_search(message)
    context = '\n'.join([doc.page_content for doc in documents])
    return context

def initialize_app(openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k")
    config = RailsConfig.from_path("config")
    app = LLMRails(config=config, llm=llm)
    return app

def format_messages(message, relevant_chunks):
    messages = [{"role": "context", "content": {"relevant_chunks": relevant_chunks}}, {"role": "user", "content": message}]
    return messages

async def predict(message, _, openai_api_key):
    if not openai_api_key:
        return "OpenAI API Key is required to run this demo, please enter your OpenAI API key in the settings and configs section!"
    
    app = initialize_app(openai_api_key)
    context = vector_search(openai_api_key, message)
    response = await app.generate_async(messages=format_messages(message, context))
    return response["content"]


examples = [["How to host my website in AWS?"], ["Tell me about AWS CodeStar?"]]
additional_inputs = [gr.Textbox(placeholder="Enter your OpenAI API key", type="password", value=OPENAI_API_KEY, show_label=False)]
chat_textbox = gr.Textbox(placeholder="Hello, Ask any question related to AWS", container=False, scale=6)
additional_inputs_accordion = gr.Accordion(label="Settings and Configs", open=True)

demo = gr.ChatInterface(predict, chatbot=gr.Chatbot(height=500), 
                        textbox=chat_textbox, title="AWS Chatbot | Guardrails",  examples=examples,
                        description="Experiment on langchain with NeMo Guardrails", theme="soft", 
                        additional_inputs_accordion=additional_inputs_accordion, additional_inputs=additional_inputs)

if __name__ == "__main__":
    demo.launch()
