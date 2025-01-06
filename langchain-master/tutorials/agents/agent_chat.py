'''
@Project ：langchain-master 
@File    ：agent_chat.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/6 15:33 
'''
import asyncio
import getpass
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from utils import set_proxy

if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass()
if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


def get_tools():
    search = TavilySearchResults(max_results=2)
    tools = [search]

    return tools

def get_model():
    model = ChatOpenAI(model="gpt-4")
    return model

def chat_without_tool():
    model = get_model()
    response = model.invoke([HumanMessage(content="hi!")])
    print(f"response:{response}")
    print("-------------------------------")
    print(f"content of response:{response.content}")

def chat_with_tool():
    model = get_model()
    tools = get_tools()
    model_with_tools = model.bind_tools(tools)
    # -------------------------------------------
    response_1 = model_with_tools.invoke([HumanMessage(content="Hi!")])
    print(f"ContentString: {response_1.content}")
    print(f"ToolCalls: {response_1.tool_calls}")

    print("################################################")
    response_2 = model_with_tools.invoke([HumanMessage(content="What's the weather in SF?")])
    print(f"ContentString: {response_2.content}")
    print(f"ToolCalls: {response_2.tool_calls}")

def get_agent():
    model = get_model()
    tools = get_tools()

    agent_executor = create_react_agent(model, tools)
    return agent_executor


def run_agent():
    agent_executor = get_agent()
    print("--1-----------------------------------------")
    response = agent_executor.invoke({"messages": [HumanMessage(content="hi!")]})
    print(f"message of response:{response["messages"]}")

    print("--2------------------------------------")
    response = agent_executor.invoke(
        {"messages": [HumanMessage(content="whats the weather in sf?")]}
    )
    print(f"message of response:{response["messages"]}")
    print("--3------------------------------------")
    for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content="whats the weather in sf?")]}
    ):
        print(chunk)
        print("----")

async def async_chat_with_agent():
    agent_executor = get_agent()
    async for event in agent_executor.astream_events(
            {"messages": [HumanMessage(content="whats the weather in sf?")]}, version="v1"
    ):
        kind = event["event"]
        if kind == "on_chain_start":
            if (
                    event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print(
                    f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                )
        elif kind == "on_chain_end":
            if (
                    event["name"] == "Agent"
            ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                print()
                print("--")
                print(
                    f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                )
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")
        elif kind == "on_tool_start":
            print("--")
            print(
                f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
            )
        elif kind == "on_tool_end":
            print(f"Done tool: {event['name']}")
            print(f"Tool output was: {event['data'].get('output')}")
            print("--")

if __name__ == '__main__':
    set_proxy()

    if True:
        asyncio.run(async_chat_with_agent())
    else:
        chat_without_tool()
        chat_with_tool()
        run_agent()