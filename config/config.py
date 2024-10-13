from nemoguardrails import LLMRails
from nemoguardrails.actions.actions import ActionResult
from vectorstore import vector_search

def prompt_template(question, context):
    return f"""You are an **GitDoc AI** Chatbot, a helpful assistant that assists users with their 
    **NVIDIA's NeMo Guardrails** related questions.
    CONTEXT INFORMATION is below.
    ---------------------
    {context}
    ---------------------

    RULES:
    1. Only Answer the USER QUESTION using the CONTEXT INFORMATION text above.
    2. Keep your answer grounded in the facts of the CONTEXT. 
    3. If you don't know the answer, just say that you don't know politely.
    4. Should not answer any out-of-context USER QUESTION.
    5. Provide only complete source url as references in markdown format.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

def rag(context: dict, llm) -> ActionResult:
    context_updates = {}
    user_message = context.get("last_user_message")
    relevant_chunks = vector_search(user_message)
    context_updates["relevant_chunks"] = relevant_chunks
    prompt = prompt_template(user_message, relevant_chunks)
    answer = llm.invoke(prompt).content
    context_updates["_last_bot_prompt"] = prompt
    return ActionResult(return_value=answer, context_updates=context_updates)

def init(app: LLMRails):
    app.register_action(rag, "rag")