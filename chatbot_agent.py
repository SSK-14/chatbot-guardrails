import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
import pickle
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

with open("vectorstore/gpt-4.pkl", 'rb') as f: 
    vectorstore = pickle.load(f)

def predict(message, history):
    llm = ChatOpenAI(temperature=0)
    gpt_4_information = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever()
    )
    tools = [
        Tool(
            name="Information about GPT-4",
            func=gpt_4_information.run,
            description="useful for when you need to answer questions about the GPT-4. Input should be a fully formed question.",
        ),
    ]
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    ai_response = agent.run(message)
    return ai_response

demo = gr.ChatInterface(predict, chatbot=gr.Chatbot(height=600), examples=["Tell me about GPT-4", "Why should we use GPT-4"], 
                        textbox=gr.Textbox(placeholder="Ask any question related to GPT-4", container=False, scale=7), title="GPT-4 Bot", 
                        description="Experiment on langchain agent tools with vectorstore", theme="soft",)

if __name__ == "__main__":
    demo.launch()