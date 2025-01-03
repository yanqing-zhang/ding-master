'''
@Project ：langchain-master 
@File    ：async_chatbots.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/3 13:39 
'''
import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
import asyncio

def get_model():
  model = ChatOpenAI(model="gpt-4o-mini")
  return model


# Async function for node:
async def call_model(state: MessagesState):
    model = get_model()
    response = await model.ainvoke(state["messages"])
    return {"messages": response}

def get_workflow():
    # Define graph as before:
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)
    app = workflow.compile(checkpointer=MemorySaver())
    return app

async def chat():
    """
    实现异步使用call_model
    """
    app = get_workflow()
    config = {"configurable": {"thread_id": "abc123"}}
    query = "Hi! I'm Bob."
    input_messages = [HumanMessage(query)]

    # Async invocation:
    output = await app.ainvoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()

if __name__ == '__main__':
    asyncio.run(chat())