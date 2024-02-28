import os
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from nemoguardrails import LLMRails, RailsConfig
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from create_index import VECTOR_DB_PATH, embedding
import gradio as gr
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

prompt_template = """You are an intelligent chatbot helping the user with Amazon Web Service (AWS) related questions. Use the following pieces of context about AWS to answer the question at the end. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
### Context: {context} ###
User Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def get_qa_chain(llm):
    chain_type_kwargs = {"prompt": PROMPT}
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(), chain_type_kwargs=chain_type_kwargs, verbose=True)
    return qa_chain

llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model_name="gpt-3.5-turbo-0125")
vector_db = Chroma(persist_directory=VECTOR_DB_PATH, embedding_function=embedding)

config = RailsConfig.from_path("config")
app = LLMRails(config=config, llm=llm)
qa_chain = get_qa_chain(app.llm)
app.register_action(qa_chain, name="qa_chain")

def messages(message, history):
    messages = []
    for conversation in history:
        messages.append({"role": "user", "content": conversation[0]})
        messages.append({"role": "bot", "content": conversation[1]})
    messages.append({"role": "user", "content": message})
    return messages

async def predict(message, history):
    bot_message = await app.generate_async(messages=messages(message, history))
    return bot_message["content"]

demo = gr.ChatInterface(predict, chatbot=gr.Chatbot(height=500), examples=["How to host my website in aws?", "Tell me about AWS CodeStar?"], 
                        textbox=gr.Textbox(placeholder="Ask any question related to AWS", container=False, scale=6), title="AWS Bot", 
                        description="Experiment on langchain with NeMo Guardrails", theme="soft",)

if __name__ == "__main__":
    demo.launch()