import os
import gradio as gr
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
from chain import qa_chain, initialize_llm
from vectorstore import vector_search
from ui import chat, chat_examples, custom_css, header

# Load environment variables
load_dotenv()

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
MODEL_API_KEY = os.getenv("MODEL_API_KEY") or ""

# Define model list with providers
MODEL_LIST = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-1.5-pro-002",
    "ollama": "llama3.2",
    "groq": "llama-3.2-11b-text-preview"
}
DEFAULT_MODEL = "openai"  # Set default model provider

# Initialize application with LLMRails
def initialize_app(llm):
    config = RailsConfig.from_path("config")
    app = LLMRails(config=config, llm=llm)
    return app

# Prediction function to generate responses
async def predict(message, _, model_api_key, provider, is_guardrails):
    if not model_api_key:
        return "OpenAI/Gemini/Groq API Key is required to run this demo. Please enter your OpenAI API key in the settings and configs section!"
    
    llm = initialize_llm(model_api_key, provider, MODEL_LIST[provider])
    
    if not is_guardrails:
        context = vector_search(message)
        return qa_chain(llm, message, context)
    else:
        app = initialize_app(llm)
        messages = [{"role": "user", "content": message}]
        options = {"output_vars": ["triggered_input_rail", "triggered_output_rail"]}
        output = await app.generate_async(messages=messages, options=options)
        print(output.output_data)
        return output.response[0]['content']

# Gradio UI setup
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""<div style='height: 10px'></div>""")
    
    with gr.Row():
        with gr.Column(scale=1):
            header()
        with gr.Column(scale=2):
            with gr.Group():
                with gr.Row():
                    guardrail = gr.Checkbox(label="Enable NeMo Guardrails", value=True, scale=1)
                    provider = gr.Dropdown(MODEL_LIST.keys(), value=DEFAULT_MODEL, show_label=False, scale=2)
                    model_key = gr.Textbox(
                        placeholder="Enter your OpenAI/Gemini/Groq API key", 
                        type="password", 
                        value=MODEL_API_KEY, 
                        show_label=False, 
                        scale=4
                    )
    
    gr.ChatInterface(
        predict,
        chatbot=chat(),
        examples=chat_examples,
        type="messages",
        additional_inputs=[model_key, provider, guardrail]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch()
