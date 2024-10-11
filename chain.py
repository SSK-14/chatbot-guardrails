def prompt_template(question, context):
    return f"""You are an Amazon Web Service (AWS) Chatbot, a helpful assistant that assists users with their AWS-related questions. 
    Use the following pieces of context to answer the user's question:
    {context}

    USER QUESTION: ```{question}```
    Answer in markdown:"""

def qa_chain(llm, message, context):
    return llm.invoke(prompt_template(message, context)).content