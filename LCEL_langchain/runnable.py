from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

prompt = ChatPromptTemplate.from_template(
    """
         tell me curois facts about {soccer_player}
    """
    )
llm = ChatGroq(
    model='qwen-2.5-32b',
    api_key=os.getenv('GROQ_API_KEY'),
)

output_parser = StrOutputParser()
# Create a chain with the prompt and the parser this pipe (|) is called a Runnble in LCEL
chain = prompt| llm | output_parser

res = chain.invoke({'soccer_player':'Lionel Messi'})
print(res)