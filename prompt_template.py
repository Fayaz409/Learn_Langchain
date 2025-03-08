from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os
load_dotenv()

llm = ChatGroq(
    model='qwen-2.5-32b',
    api_key=os.getenv('GROQ_API_KEY'),

)

prompt_template = PromptTemplate.from_template(
    """
    You are a Math teacher and always teach with best possible and fun way. Solve this {question} in perfect and poetic way"""
)

llm_prompt = prompt_template.format(
    question="What is the square root of 25?"
)
print('OUTPUT:')
print(llm.invoke(llm_prompt))

