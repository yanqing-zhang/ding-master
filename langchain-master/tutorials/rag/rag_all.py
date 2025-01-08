'''
@Project ：langchain-master 
@File    ：rag_all.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/8 9:37 
'''
import getpass
import os

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langgraph.graph import MessagesState, StateGraph
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# 设置agent代理
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'


if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
# ~------------------------------------------------------------------

def get_model():
    llm = ChatOpenAI(model="gpt-4o-mini")
    return llm

def get_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings

def get_vector_db():
    embeddings = get_embeddings()
    vector_store = InMemoryVectorStore(embeddings)
    return vector_store

def get_docs():
    # Load and chunk contents of the blog
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    return docs

def get_split():
    vector_store = get_vector_db()
    docs = get_docs()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)

    # Index chunks
    _ = vector_store.add_documents(documents=all_splits)

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    vector_store = get_vector_db()
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Step 1: Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm = get_model()
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

def get_tools():
    # Step 2: Execute the retrieval.
    tools = ToolNode([retrieve])
    return tools


# Step 3: Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""
    # Get generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        f"{docs_content}"
    )
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_content)] + conversation_messages

    # Run
    llm = get_model()
    response = llm.invoke(prompt)
    return {"messages": [response]}

def get_graph_builder():
    graph_builder = StateGraph(MessagesState)
    return graph_builder
def get_graph():
    tools = get_tools()

    graph_builder = get_graph_builder()
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)

    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    graph = graph_builder.compile()
    return graph


def show_graph():
    graph = get_graph()
    display(Image(graph.get_graph().draw_mermaid_png()))


def get_memory():
    graph_builder = get_graph_builder()
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)


def get_agent():
    llm = get_model()
    memory = get_memory()
    agent_executor = create_react_agent(llm, [retrieve], checkpointer=memory)
    display(Image(agent_executor.get_graph().draw_mermaid_png()))

    config = {"configurable": {"thread_id": "def234"}}

    input_message = (
        "What is the standard method for Task Decomposition?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )

    for event in agent_executor.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
        config=config,
    ):
        event["messages"][-1].pretty_print()


# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}
def chat():
    graph = get_graph()
    input_message = "Hello"

    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    input_message = "What is Task Decomposition?"

    for step in graph.stream(
        {"messages": [{"role": "user", "content": input_message}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    # ------------------------------------

    for step in graph.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
    ):
        step["messages"][-1].pretty_print()


if __name__ == '__main__':
    # chat()
    get_agent()