import os, asyncio
import gradio as gr
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
from chain import initialize_llm, rag_chain
from ui import chat, demo_header_settings, custom_css, chat_examples

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_LIST = {
    "openai": "gpt-4o-mini",
    "groq": "llama-3.2-11b-text-preview",
    "gemini": "gemini-1.5-pro-002",
}

def init_app(api_key, provider):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        llm = initialize_llm(api_key, provider, MODEL_LIST[provider])
        config = RailsConfig.from_path("nemo")
        app = LLMRails(config=config, llm=llm)
        gr.Info(f"Chat initialized with {provider}")
        return app, llm
    except Exception as e:
        gr.Error(f"Error initializing the app: {e}")
        return None, None

# Prediction function to generate responses
def predict(message, history, app, llm, is_guardrails=True):
    if not app or not llm:
        return "Chatbot not initialized. Please start chat first."
    if is_guardrails:
        history.append({"role": "user", "content": message})
        options = {"output_vars": ["triggered_input_rail", "triggered_output_rail"]}
        output = app.generate(messages=history, options=options)
        info = app.explain()
        info.print_llm_calls_summary()
        warning_message = output.output_data["triggered_input_rail"] or output.output_data["triggered_output_rail"]
        if warning_message:
            gr.Warning(f"Guardrail triggered: {warning_message}")
        return output.response[0]['content']
    else:
        return rag_chain(llm, message)

def respond(message, chat_history, app, llm, guardrail_enabled):
    bot_message = predict(message, chat_history, app, llm, guardrail_enabled)
    chat_history.append({"role": "assistant", "content": bot_message})
    return "", chat_history


# Gradio UI setup
with gr.Blocks(css=custom_css) as demo:
    app_state = gr.State(None)
    llm_state = gr.State(None)
    model_key, provider, guardrail, start_chat = demo_header_settings(MODEL_LIST)
    start_chat.click(
        init_app, 
        [model_key, provider], 
        [app_state, llm_state]
    )
    chatbot = chat()
    msg = gr.Textbox(placeholder="Type your message here...", type="text", show_label=False, submit_btn=True)
    examples = gr.Examples(chat_examples, msg)
    msg.submit(
        respond, 
        [msg, chatbot, app_state, llm_state, guardrail], 
        [msg, chatbot]
    )

# Launch the application
if __name__ == "__main__":
    demo.launch()
