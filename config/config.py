from nemoguardrails import LLMRails
from nemoguardrails.actions.actions import ActionResult

def prompt_template(question, context):
    return f"""You are an **GitDoc AI** Chatbot, a helpful assistant that assists users with their **NVIDIA's NeMo Guardrails** related questions.
    CONTEXT INFORMATION is below.
    ---------------------
    {context}
    ---------------------

    RULES:
    1. Only Answer the USER QUESTION using the CONTEXT text above.
    2. Keep your answer grounded in the facts of the CONTEXT. 
    3. If you don't know the answer, just say that you don't know.
    4. Should not answer any out-of-context USER QUESTION.
    5. Provide right references if needed.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

def rag(context: dict, llm) -> ActionResult:
    user_message = context.get("last_user_message")
    relevant_chunks = context.get("relevant_chunks")
    context_updates = {}

    answer = llm.invoke(prompt_template(user_message, relevant_chunks)).content
    return ActionResult(return_value=answer, context_updates=context_updates)

def init(app: LLMRails):
    app.register_action(rag, "rag")