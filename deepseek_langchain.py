#pip install -qU langchain-deepseek-official
from langchain_deepseek import ChatDeepSeek
from decouple import config
DEBUG = config('DEBUG', default=False, cast=bool)


llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=config('DEEPSEEK_API_KEY', default='fallback_secret')
    
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
 
llm.invoke(messages)

for chunk in llm.stream(messages):
    print(chunk)