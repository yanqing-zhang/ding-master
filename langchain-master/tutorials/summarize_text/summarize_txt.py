'''
@Project ：langchain-master 
@File    ：summarize_txt.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/10 9:33 
'''
import asyncio
import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain_text_splitters import CharacterTextSplitter
import matplotlib.pyplot as plt
import matplotlib.image as mpimg  # 导入matplotlib.image用于读取图像
import operator
from typing import Annotated, List, Literal, TypedDict

from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from langchain_core.documents import Document
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


def get_docs():
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()
    return docs

def get_model():
    llm = ChatOpenAI(model="gpt-4o-mini")
    return llm

def get_prompt():
    # Define prompt
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of the following:\\n\\n{context}")]
    )
    return prompt

def get_chain():
    llm = get_model()
    prompt = get_prompt()
    # Instantiate chain
    chain = create_stuff_documents_chain(llm, prompt)
    return chain

def chain_test():
    chain = get_chain()
    docs = get_docs()
    # Invoke chain
    result = chain.invoke({"context": docs})
    print(result)

    for token in chain.stream({"context": docs}):
        print(token, end="|")



def get_map_prompt():
    map_prompt = ChatPromptTemplate.from_messages(
        [("system", "Write a concise summary of the following:\\n\\n{context}")]
    )
    map_prompt = hub.pull("rlm/map-prompt")
    return map_prompt

def get_reduce_prompt():
    # Also available via the hub: `hub.pull("rlm/reduce-prompt")`
    reduce_template = """
    The following is a set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary
    of the main themes.
    """

    reduce_prompt = ChatPromptTemplate([("human", reduce_template)])
    return reduce_prompt


def get_split_docs():
    docs = get_docs()
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Generated {len(split_docs)} documents.")
    return split_docs

def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    llm = get_model()
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)


# This will be the overall state of the main graph.
# It will contain the input document contents, corresponding
# summaries, and a final summary.
class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]
    final_summary: str


# This will be the state of the node that we will "map" all
# documents to in order to generate summaries
class SummaryState(TypedDict):
    content: str


# Here we generate a summary, given a document
async def generate_summary(state: SummaryState):
    llm = get_model()
    map_prompt = get_map_prompt()
    prompt = map_prompt.invoke(state["content"])
    response = await llm.ainvoke(prompt)
    return {"summaries": [response.content]}


# Here we define the logic to map out over the documents
# We will use this an edge in the graph
def map_summaries(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]


def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }


async def _reduce(input: dict) -> str:
    llm = get_model()
    reduce_prompt = get_reduce_prompt()
    prompt = reduce_prompt.invoke(input)
    response = await llm.ainvoke(prompt)
    return response.content


# Add node to collapse summaries
async def collapse_summaries(state: OverallState):
    token_max = 1000
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, _reduce))

    return {"collapsed_summaries": results}


# This represents a conditional edge in the graph that determines
# if we should collapse the summaries or not
def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    token_max = 1000
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"


# Here we will generate the final summary
async def generate_final_summary(state: OverallState):
    response = await _reduce(state["collapsed_summaries"])
    return {"final_summary": response}


def get_graph():
    # Construct the graph
    # Nodes:
    graph = StateGraph(OverallState)
    graph.add_node("generate_summary", generate_summary)  # same as before
    graph.add_node("collect_summaries", collect_summaries)
    graph.add_node("collapse_summaries", collapse_summaries)
    graph.add_node("generate_final_summary", generate_final_summary)

    # Edges:
    graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
    graph.add_edge("generate_summary", "collect_summaries")
    graph.add_conditional_edges("collect_summaries", should_collapse)
    graph.add_conditional_edges("collapse_summaries", should_collapse)
    graph.add_edge("generate_final_summary", END)

    app = graph.compile()
    return graph, app



def show():
    graph, _ = get_graph()
    try:
        # 使用 Mermaid 生成图表并保存为文件
        mermaid_code = graph.get_graph().draw_mermaid_png()
        with open("graph.jpg", "wb") as f:
            f.write(mermaid_code)

        # 使用 matplotlib 显示图像
        img = mpimg.imread("graph.jpg")
        plt.imshow(img)
        plt.axis('off')  # 关闭坐标轴
        plt.show()
    except Exception as e:
        print(f"An error occurred: {e}")

async def chat():
    _, app = get_graph()
    split_docs = get_split_docs()
    async for step in app.astream(
        {"contents": [doc.page_content for doc in split_docs]},
        {"recursion_limit": 10},
    ):
        print(list(step.keys()))


if __name__ == '__main__':
    asyncio.run(chat())
    show()
    chain_test()