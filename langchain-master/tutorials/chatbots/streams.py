'''
@Project ：langchain-master 
@File    ：streams.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/3 16:20 
'''
import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_model():
  model = ChatOpenAI(model="gpt-4o-mini")
  return model

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

def get_workflow():
    workflow = StateGraph(state_schema=MessagesState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def chat():
    app = get_workflow()
    config = {"configurable": {"thread_id": "abc789"}}
    query = "Hi I'm Todd, please tell me a joke."
    language = "English"

    input_messages = [HumanMessage(query)]
    for chunk, metadata in app.stream(
        {"messages": input_messages, "language": language},
        config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage):  # Filter to just model responses
            print(chunk.content, end="|")

if __name__ == '__main__':
    chat()