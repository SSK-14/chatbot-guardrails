from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

PROMPT_TEMPLATE = """You are an Amazon Web Service (AWS) Chatbot, a helpful assistant that assists users with their AWS-related questions. 
Use the following pieces of context to answer the user's question:
{context}

USER QUESTION: ```{question}```
Answer in markdown:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

async def qa_chain(llm, message, context):
    prompt_template = PromptTemplate.from_template(PROMPT_TEMPLATE)
    input_variables = {"question": message, "context": context}
    chain = prompt_template | llm | StrOutputParser()
    response = await chain.ainvoke(input_variables)
    return response

