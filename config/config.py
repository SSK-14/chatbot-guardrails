from langchain.llms.base import BaseLLM
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from nemoguardrails import LLMRails
from nemoguardrails.actions.actions import ActionResult

PROMPT_TEMPLATE = """You are an Amazon Web Service (AWS) Chatbot, a helpful assistant that assists users with their AWS-related questions. Use the following pieces of context to answer the user's question:
    CONTEXT INFORMATION is below.
    ---------------------
    {context}
    ---------------------

    RULES:
    1. Only Answer the USER QUESTION using the CONTEXT text above.
    2. Keep your answer grounded in the facts of the CONTEXT. 
    3. If you don't know the answer, just say that you don't know.
    4. Should not answer any out-of-context USER QUESTION.

    USER QUESTION: ```{question}```
    Answer in markdown:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

async def rag(context: dict, llm: BaseLLM) -> ActionResult:
    user_message = context.get("last_user_message")
    relevant_chunks = context.get("relevant_chunks")
    context_updates = {}

    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    input_variables = {"question": user_message, "context": relevant_chunks}
    context_updates["_last_bot_prompt"] = prompt_template.format(**input_variables)

    output_parser = StrOutputParser()
    chain = prompt_template | llm | output_parser
    answer = await chain.ainvoke(input_variables)
    return ActionResult(return_value=answer, context_updates=context_updates)

def init(app: LLMRails):
    app.register_action(rag, "rag")