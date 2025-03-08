from langchain_groq import ChatGroq
# from dotenv import load_dotenv
import os
from decouple import config
DEBUG = config('DEBUG', default=False, cast=bool)

# load_dotenv()

llm = ChatGroq(
    model='qwen-2.5-32b',
    api_key=config('GROQ_API_KEY', default='fallback_secret'),

)

messages = [
    {
        "role":"system",
        "content": "You are a Math teacher and always teach with best possible wat in Manim Language and tell the user how to run it",

    },
    {   "role": "human",
        "content": "What is the square root of 25?"
    
    }
]

res = llm.invoke(messages)
print(res.content)