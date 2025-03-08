from langchain_core.prompts import PromptTemplate,FewShotChatMessagePromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
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

print('\n')



# ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
    ("system","You are an {profession} in this topic {topic}"),
    (
        "human","Hello, Mr {profession}. Can you help me with {topic}?"
    ),
    (
        "ai","Sure"
    ),
    (
        "user","{user_input}"
    ),
    ]
 )

messages = chat_template.format_messages(
    profession="Math teacher",
    topic="Algebra",
    user_input="What is the square root of 25?"
)

print('ChatPromptTemplate OUTPUT:')
print(llm.invoke(messages))


# FewShotChatMessagePromptTemplate
print('\n')
print('FewShotChatMessagePromptTemplate OUTPUT:')

examples = [
    {"input": "hi1", "output": "iholai"},
    {"input": "bye1", "output": "iadios1"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}"),
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an English-Spanish translator."),
    few_shot_prompt,
    ("human", "{input}"),
])

f_template = final_prompt.format_messages(input="hi1")

print(llm.invoke(f_template))
# print(llm.invoke(final_prompt.format(input="hi1")))