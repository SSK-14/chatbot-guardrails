import os
import gradio as gr
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from chain import qa_chain
from vectorstore import vector_search

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GROQ_API_KEY") or ""
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

MODEL_LIST = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-pro-002",
    "ollama": "llama3.2",
    "groq": "llama-3.2-11b-text-preview"
}
DEFAULT_MODEL = "openai"

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

    if provider == "gemini":
        llm = ChatGoogleGenerativeAI(google_api_key=model_api_key, model=MODEL_LIST[provider])
    elif provider == "openai":
        llm = ChatOpenAI(openai_api_key=model_api_key, model_name=MODEL_LIST[provider])
    elif provider == "groq":
        llm = ChatOpenAI(openai_api_key=model_api_key, openai_api_base=GROQ_BASE_URL, model_name=MODEL_LIST[provider])
    elif provider == "ollama":
        llm = ChatOpenAI(openai_api_key="", openai_api_base=OLLAMA_BASE_URL, model_name=MODEL_LIST[provider])
    else:
        return "Invalid provider selected. Please select a valid provider."

    context = vector_search(message)

    if not is_guardrails:
        return qa_chain(llm, message, context)
    
    app = initialize_app(llm)
    response = await app.generate_async(messages=format_messages(message, context))
    return response["content"]

with gr.Blocks() as demo:
    gr.HTML("""<div style='height: 10px'></div>""")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                # NeMo Guardrails Chatbot 💂🏼
                Ask questions about [NVIDIA's NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/index.html) documentations.
                """
            )
        with gr.Column(scale=2):
            with gr.Group():
                with gr.Row():
                    guardrail = gr.Checkbox(label="Enable NeMo Guardrails", value=True, scale=1)
                    provider = gr.Dropdown(MODEL_LIST.keys(), value=DEFAULT_MODEL, show_label=False, scale=2)
                    model_key = gr.Textbox(placeholder="Enter your OpenAI/Gemini API key", type="password", value=MODEL_API_KEY, show_label=False, scale=4)

    gr.ChatInterface(
        predict,
        chatbot=gr.Chatbot(height=600, type="messages", layout="panel"),
        theme="soft",
        examples=[["What LLMs are supported by NeMo Guardrails ?"], ["Can I deploy NeMo Guardrails in production ?"]],
        type="messages",
        additional_inputs=[model_key, provider, guardrail]
    )

if __name__ == "__main__":
    demo.launch()
