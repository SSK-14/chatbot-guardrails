import os
import gradio as gr
from dotenv import load_dotenv
from nemoguardrails import LLMRails, RailsConfig
from chain import initialize_llm, rag_chain
from ui import chat, header, chat_examples, custom_css

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set the model provider and API key
MODEL_PROVIDER = "openai"
MODEL_NAME = "gpt-4o-mini"
MODEL_API_KEY = os.getenv("MODEL_API_KEY") or ""
IS_GUARDRAILS = True

llm = initialize_llm(MODEL_API_KEY, MODEL_PROVIDER, MODEL_NAME)
config = RailsConfig.from_path("nemo")
app = LLMRails(config=config, llm=llm)

# Prediction function to generate responses
async def predict(message, history):
    if IS_GUARDRAILS:
        history.append({"role": "user", "content": message})
        options = {"output_vars": ["triggered_input_rail", "triggered_output_rail"]}
        output = await app.generate_async(messages=history, options=options)
        info = app.explain()
        info.print_llm_calls_summary()
        warning_message = output.output_data["triggered_input_rail"] or output.output_data["triggered_output_rail"]
        if warning_message:
            gr.Warning(f"Guardrail triggered: {warning_message}")
        return output.response[0]['content']
    else:
        return rag_chain(llm, message)

# Gradio UI setup
with gr.Blocks(css=custom_css) as demo:
    header()
    gr.ChatInterface(
        predict,
        chatbot=chat(),
        examples=chat_examples,
        type="messages",
    )

# Launch the application
if __name__ == "__main__":
    demo.launch()
