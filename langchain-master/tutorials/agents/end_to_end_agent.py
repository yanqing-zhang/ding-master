'''
@Project ：langchain-master 
@File    ：end_to_end_agent.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/4 21:56 
'''
# Import relevant functionality
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

import getpass
import os

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass()

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


def get_tavil_key():
    tavily_key = os.getenv("TAVILY_API_KEY")
    print(f"tavily_key:{tavily_key}")
    return tavily_key

def get_model():
    model = ChatOpenAI(model="gpt-4o-mini")
    # model = ChatAnthropic(model_name="claude-3-sonnet-20240229")
    return model

def get_memory():
    # Create the agent
    memory = MemorySaver()
    return memory

def get_tavil_client():
    client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return client
def do_agent():
    t_key = get_tavil_key()
    print(f"t_key:{t_key}")
    print(f"===============================")
    model = get_model()
    memory = get_memory()
    search = TavilySearchResults(max_results=2)
    tools = [search]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    # Use the agent
    config = {"configurable": {"thread_id": "abc123"}}
    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="hi im bob! and i live in sf")]}, config
    ):
        print(chunk)
        print("----")

    for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="whats the weather where I live?")]}, config
    ):
        print(chunk)
        print("----")

if __name__ == '__main__':
    os.environ["http_proxy"] = "http://127.0.0.1:10792"
    os.environ["https_proxy"] = "http://127.0.0.1:10792"

    do_agent()