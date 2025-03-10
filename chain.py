from langchain_core.prompts import ChatPromptTemplate, PromptTemplate,FewShotChatMessagePromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

examples =[
      {"input": "hi", "output": "hola!"},
    {"input": "bye", "output": "adios!"},   
]

example_template = ChatPromptTemplate.from_messages(
    [
        ("human","{input}"),
        ("ai","{output}")
    ]
)

few_shot_template = FewShotChatMessagePromptTemplate(
    example_prompt=example_template,
    examples=examples
)

final_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are an English-Spanish translator."),
        few_shot_template,
        ("human","{input}")
    ]
)

llm = ChatGroq(
    model='qwen-2.5-32b',
    api_key=os.getenv('GROQ_API_KEY'),
)

chain = final_template | llm
print(chain)
# messages = chain.format_messages(input="hi1")
# print(chain.invoke({"input":"Who is Kennedy?"}))