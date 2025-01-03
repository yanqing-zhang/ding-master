'''
@Project ：langchain-master 
@File    ：manage_conversation_history.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/3 15:37 
'''
from langchain_core.messages import SystemMessage, trim_messages

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from typing import Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You talk like a pirate. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

def get_model():
  model = ChatOpenAI(model="gpt-4o-mini")
  return model

def get_trimmer():
    model = get_model()
    trimmer = trim_messages(
        max_tokens=65,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )
    return trimmer

def get_messages():
    messages = [
        SystemMessage(content="you're a good assistant"),
        HumanMessage(content="hi! I'm bob"),
        AIMessage(content="hi!"),
        HumanMessage(content="I like vanilla ice cream"),
        AIMessage(content="nice"),
        HumanMessage(content="whats 2 + 2"),
        AIMessage(content="4"),
        HumanMessage(content="thanks"),
        AIMessage(content="no problem!"),
        HumanMessage(content="having fun?"),
        AIMessage(content="yes!"),
    ]
    return messages

def trim_invoke():
    trimmer = get_trimmer()
    messages = get_messages()
    result = trimmer.invoke(messages)
    print(f"result:{result}")


def get_workflow():
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def call_model(state: State):
    model = get_model()
    trimmer = get_trimmer()
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke(
        {"messages": trimmed_messages, "language": state["language"]}
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


def chat():
    app = get_workflow()
    messages = get_messages()

    config = {"configurable": {"thread_id": "abc567"}}
    query = "What is my name?"
    language = "English"

    input_messages = messages + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    output["messages"][-1].pretty_print()

    config = {"configurable": {"thread_id": "abc678"}}
    query = "What math problem did I ask?"
    language = "English"

    input_messages = messages + [HumanMessage(query)]
    output = app.invoke(
        {"messages": input_messages, "language": language},
        config,
    )
    output["messages"][-1].pretty_print()

if __name__ == '__main__':
    chat()