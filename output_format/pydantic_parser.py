from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import PromptTemplate,FewShotChatMessagePromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_groq import GroqEmbedding
import os
load_dotenv()

llm = ChatGroq(
    model='qwen-2.5-32b',
    api_key=os.getenv('GROQ_API_KEY'),

)

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
    
# Set up a parser
parser = JsonOutputParser(pydantic_object=Joke)

# Inject parser instructions into the prompt template.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Create a chain with the prompt and the parser
chain = prompt | llm | parser

response = chain.invoke({"query": "Tell me a joke about US President Donald Trump."})

print("Jokey joke:")
print(response)

print("\n----------\n")