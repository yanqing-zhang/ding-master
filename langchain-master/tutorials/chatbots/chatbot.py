import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

def get_model():
  model = ChatOpenAI(model="gpt-4o-mini")
  return model

def chat():
  model = get_model()
  r1 = model.invoke([HumanMessage(content="Hi! I'm Bob")])
  print(f"r1:{r1}")
  print("--------------------------------")
  r2 = model.invoke([HumanMessage(content="What's my name?")])
  print(f"r2:{r2}") # 因为没有上下文记忆，所以回答不出姓名是什么
  print("=================")
  r3 = model.invoke(
      [
          HumanMessage(content="Hi! I'm Bob"),
          AIMessage(content="Hello Bob! How can I assist you today?"),
          HumanMessage(content="What's my name?"),
      ]
  )
  print(f"r3:{r3}") # 因为有上下文，所以可以回答出名字为Bob



def get_workflow():
    # Define a new graph
    workflow = StateGraph(state_schema=MessagesState)
    return workflow


# Define the function that calls the model
def call_model(state: MessagesState):
    model = get_model()
    response = model.invoke(state["messages"])
    return {"messages": response}

def get_apps():
    workflow = get_workflow()
    # Define the (single) node in the graph
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # Add memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app

def app_test():
    app = get_apps()
    config = {"configurable": {"thread_id": "abc123"}}
    query = "Hi! I'm Bob."

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()  # output contains all messages in state
    print("#######################################################")
    query = "What's my name?"

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print() #可以通过上下文知道姓名为bob，因为上下文同在一个配置的thread_id：abc123里
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    config = {"configurable": {"thread_id": "abc234"}}

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print() #不知道姓名，因为不在同一个上下文中，也就是不在同一个thread_id里
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    config = {"configurable": {"thread_id": "abc123"}}

    input_messages = [HumanMessage(query)]
    output = app.invoke({"messages": input_messages}, config)
    output["messages"][-1].pretty_print()
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

if __name__ == '__main__':
  if True:
    app_test()
  else:
    chat()