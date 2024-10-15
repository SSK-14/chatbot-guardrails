import gradio as gr

# Placeholder for chat UI
chat_placeholder = """
<br>
<h1 style="text-align: center; background: black; border: 2px solid #f97316; padding: 10px 20px; border-radius: 15px;">
    Welcome to NeMo Guardrails Documentation AI
</h1>
<p style="text-align: center; margin: 10px auto; max-width: 400px">
    Explore seamless integration of GitHub repositories and documentation with LLM-powered assistance, enhanced by NeMo Guardrails for advanced safety and security.
    <br>
    💻 <a href="https://github.com/SSK-14/chatbot-guardrails target="_blank">View Code</a>
</p>
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
.examples {
    display: None;
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
        avatar_images=("./images/user.png", "./images/ai.png")
    )

# Function to render the header
def header():
    return gr.Markdown(
        """
        # NeMo Guardrails Chatbot 💂🏼
        Ask questions about [NVIDIA's NeMo Guardrails](https://docs.nvidia.com/nemo/guardrails/index.html) documentations.
        """
    )
