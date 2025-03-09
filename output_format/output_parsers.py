from langchain_core.prompts import PromptTemplate,FewShotChatMessagePromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
import os
load_dotenv()

llm = ChatGroq(
    model='qwen-2.5-32b',
    api_key=os.getenv('GROQ_API_KEY'),

)

# llmModel = OpenAI()

# from langchain_openai import ChatOpenAI

# chatModel = ChatOpenAI(model="gpt-3.5-turbo-0125")


from langchain.output_parsers.json import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)

json_parser = SimpleJsonOutputParser()

json_chain = json_prompt | llm | json_parser

response = json_chain.invoke({"question": "What is the biggest country?"})

print("What is the biggest country?")
print(response)

print("\n----------\n")