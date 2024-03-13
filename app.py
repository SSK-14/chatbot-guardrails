import os
import gradio as gr
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from nemoguardrails import LLMRails, RailsConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

VECTOR_DB_PATH = "./temp/vectorstore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or ""

PROMPT_TEMPLATE = """You are an Amazon Web Service (AWS) Chatbot, a helpful assistant that assists users with their AWS-related questions. Use the following pieces of context to answer the user's question:
    CONTEXT INFORMATION is below.
    ---------------------
    {context}
    ---------------------

    RULES:
    1. Only Answer the USER QUESTION using the CONTEXT text above.
    2. Keep your answer grounded in the facts of the CONTEXT. 
    3. If you don't know the answer, just say that you don't know.
    4. Should not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)


def get_qa_chain(llm, vector_db):
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        verbose=True,
    )
    return qa_chain


def initialize(openai_api_key):
    if not openai_api_key:
        return "OpenAI API Key is required to run this demo, please enter your OpenAI API key in the settings and configs section!"
    
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo-16k")
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key, disallowed_special=())
    vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding)
    config = RailsConfig.from_path("config")
    app = LLMRails(config=config, llm=llm)
    qa_chain = get_qa_chain(app.llm, vector_db)
    app.register_action(qa_chain, name="qa_chain")
    return app


def format_messages(message, history):
    messages = [{"role": "user", "content": content[0]} for content in history]
    messages.append({"role": "user", "content": message})
    return messages


def predict(message, history, openai_api_key):
    app = initialize(openai_api_key)
    return app.generate(messages=format_messages(message, history))["content"]


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
