from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableLambda,RunnableParallel
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

def russian(name):
    return f"Привет, {name}!"


# RunnablePassthrough is a Runnable that just passes the input to the next Runnable
# RunnaleLambda is a Runnable that runs a lambda function

# RunnableParallel is a Runnable that runs multiple Runnables in parallel
chain = RunnableParallel(
    {
       "operation_a":RunnablePassthrough(),
       "operation_b":RunnableLambda(russian), 
    }
    
)
res = chain.invoke('Lionel Messi')
print(res)

