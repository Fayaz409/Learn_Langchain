from langchain_core.messages import SystemMessage, AIMessage,HumanMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
load_dotenv()

model = ChatGroq(
    model='qwen-2.5-32b',
    api_key=os.getenv('GROQ_API_KEY'),
)

workflow = StateGraph(state_schema=MessagesState)

def extract_ai_messages(response):
    return [msg.content for msg in response["messages"] if isinstance(msg, AIMessage)]

# Define the function that calls the model
def call_model(state: MessagesState):
    system_prompt = (
        "You are a helpful assistant. "
        "Answer all questions to the best of your ability."
    )
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model.invoke(messages)
    return {"messages": response}


# Define the node and edge
workflow.add_node("model", call_model)
workflow.add_edge(START, "model")

# Add simple in-memory checkpointer
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

res_f=app.invoke(
    {"messages": [HumanMessage(content="Translate to French: I love programming.")]},
    config={"configurable": {"thread_id": "1"}},
)

res_c=app.invoke(
    {"messages": [HumanMessage(content="What did I just ask you?")]},
    config={"configurable": {"thread_id": "1"}},
)

print('French\n',res_f)
print('Context\n',res_c)
# Output:   
# Print only AI responses
print("French:", extract_ai_messages(res_f))
print("Context:", extract_ai_messages(res_c))