import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from nemoguardrails import LLMRails, RailsConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

VECTOR_DB_PATH = "./temp/vectorstore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt_template = """You are a Amazon Web Service (AWS) Chatbot an helpful assistant that helps users with their AWS related questions. Use the following pieces of context to answer the USER QUESTION.
    CONTEXT INFORMATION is below.
    ---------------------
    {context}
    ---------------------

    RULES:
    1. Only Answer the USER QUESTION using the CONTEXT text above.
    2. Keep your answer ground in the facts of the CONTEXT. 
    3. If you don't know the answer, just say that you don't know.
    4. Should not answer any out of context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_qa_chain(llm, vector_db):
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(), chain_type_kwargs=chain_type_kwargs, verbose=True)
    return qa_chain

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo-16k")
embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"), disallowed_special=())
vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding)
config = RailsConfig.from_path("config")
app = LLMRails(config=config, llm=llm)
qa_chain = get_qa_chain(app.llm, vector_db)
app.register_action(qa_chain, name="qa_chain")

def messages(message, history):
    messages = []
    for conversation in history:
        messages.append({"role": "user", "content": conversation[0]})
        messages.append({"role": "bot", "content": conversation[1]})
    messages.append({"role": "user", "content": message})
    return messages

def predict(message, history):
    bot_message = app.generate(messages=messages(message, history))
    return bot_message["content"]

demo = gr.ChatInterface(predict, chatbot=gr.Chatbot(height=500), examples=["How to host my website in aws?", "Tell me about AWS CodeStar?"], 
                        textbox=gr.Textbox(placeholder="Hello {user_info}, Ask any question related to AWS", container=False, scale=6), title="AWS Chatbot | Guardrails", 
                        description="Experiment on langchain with NeMo Guardrails", theme="soft",)

if __name__ == "__main__":
    demo.launch()