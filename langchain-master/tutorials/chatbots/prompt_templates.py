'''
@Project ：langchain-master 
@File    ：prompt_templates.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/3 14:16 
'''
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

def get_model():
  model = ChatOpenAI(model="gpt-4o-mini")
  return model

def get_workflow():
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def call_model(state: MessagesState):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You talk like a pirate. Answer all questions to the best of your ability.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    model = get_model()
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}

def chat_without_template():
    app = get_workflow()
    config = {"configurable": {"thread_id": "abc345"}}
    query = "Hi! I'm Jim."

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    print("----X-----------------------------------------")
    query = "What is my name?"

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    print("----Y-----------------------------------------")

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

def call_model_params(state: State):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    model = get_model()
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": [response]}

def get_workflow_app():
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model_params)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app




def chat_with_template():
    app = get_workflow_app()
    config = {"configurable": {"thread_id": "abc456"}}
    query = "Hi! I'm Bob."
    language = "Spanish"

    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    output["messages"][-1].pretty_print()
    print("--1-------------------------------------------")
    query = "What is my name?"

    input_messages = [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages},
        config,
    )
    output["messages"][-1].pretty_print()
    print("---2------------------------------------------")


if __name__ == '__main__':
    if True:
        chat_with_template()
    else:
        chat_without_template()