import gradio as gr

# Placeholder for chat UI
chat_placeholder = """
<div style="margin: 40px 20px;">
<h1 style="text-align: center; background: #18181c; border: 2px solid #f97316; padding: 10px 20px; border-radius: 15px;">
    Welcome to NeMo Guardrails Documentation AI
</h1>
<p style="font-size: 16px; text-align: center; margin: 10px auto; max-width: 650px">
    Explore seamless integration of GitHub repositories and documentation with LLM-powered assistance, enhanced by NeMo Guardrails for advanced safety and security.
</p>
</div>
"""

# Chat examples for user guidance
chat_examples = [
    ["What LLMs are supported by NeMo Guardrails ?"],
    ["Can I deploy NeMo Guardrails in production ?"]
]

# Custom CSS for styling
custom_css = """
a {
    color: #f97316;
}
.avatar-image {
    margin: 0;
}
"""

# Function to create a chatbot with custom settings
def chat():
    return gr.Chatbot(
        height=600, 
        type="messages", 
        elem_classes="chatbot", 
        placeholder=chat_placeholder, 
        layout="panel", 
        avatar_images=("./images/user.png", "./images/ai.png"),
    )

# Function to render the header
def header():
    return gr.Markdown(
        """
        # NeMo Guardrails Chatbot 💂🏼
        Ask questions about [NVIDIA's NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/index.html) documentations.
        """
    )

def demo_header_settings(model_list):
    gr.HTML("""<div style='height: 10px'></div>""")
    with gr.Row():
        with gr.Column(scale=1):
            header()
        with gr.Column(scale=2):
            with gr.Row():
                guardrail = gr.Checkbox(label="Enable NeMo Guardrails", value=True, scale=1)
                provider = gr.Dropdown(list(model_list.keys()), value=list(model_list.keys())[0], show_label=False, scale=2)
                model_key = gr.Textbox(
                    placeholder="Enter your OpenAI/Gemini/Groq API key", 
                    type="password",
                    show_label=False, 
                    scale=4
                )
                start_chat = gr.Button("Initialize Chat")
        return model_key, provider, guardrail, start_chat