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

# chain.invoke()
# chain.stream()
# chain.batch()

output_parser = StrOutputParser()

chain = prompt| llm | output_parser

invoke_res = chain.invoke({'soccer_player':'Lionel Messi'})
print('Invoke\n',invoke_res)

for stream_res in chain.stream({'soccer_player':'Lionel Messi'}):
    print(stream_res, end='\n\n',flush=True)

batch_res = chain.batch([{'soccer_player':'Lionel Messi'},{'soccer_player':'Cristiano Ronaldo'}])
print('Batch response\n',batch_res)


# there are also Asynchronous versions of the above methods
# chain.ainvoke()
# chain.astream()
# chain.abatch()