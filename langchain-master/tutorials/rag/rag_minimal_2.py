'''
@Project ：langchain-master 
@File    ：rag_minimal.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/6 16:11 
'''
import getpass
import os

os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from utils import set_proxy
from IPython.display import Image, display
from typing import Literal
from typing_extensions import Annotated

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

if not os.environ.get("LANGCHAIN_API_KEY"):
    # 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
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Enter API key for LangChain Smith: ")

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

def split_docs():
    docs = get_docs()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def split_phases():
    all_splits = split_docs()
    total_documents = len(all_splits)
    third = total_documents // 3

    for i, document in enumerate(all_splits):
        if i < third:
            document.metadata["section"] = "beginning"
        elif i < 2 * third:
            document.metadata["section"] = "middle"
        else:
            document.metadata["section"] = "end"
    print(f"metadata:{all_splits[0].metadata}")


def save():
    all_splits = split_docs()
    vector_store = get_vector_db()
    # Index chunks
    document_ids = vector_store.add_documents(documents=all_splits)
    print(f"document_ids:{document_ids[:3]}")

def get_prompt():
    # Define prompt for question-answering
    prompt = hub.pull("rlm/rag-prompt")

    example_messages = prompt.invoke(
        {"context": "(context goes here)", "question": "(question goes here)"}
    ).to_messages()

    assert len(example_messages) == 1
    print(example_messages[0].content)
    return prompt

class Search(TypedDict):
    """Search query."""

    query: Annotated[str, ..., "Search query to run."]
    section: Annotated[
        Literal["beginning", "middle", "end"],
        ...,
        "Section to query.",
    ]

# Define state for application
class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


class State(TypedDict):
    question: str
    query: Search
    context: List[Document]
    answer: str


def analyze_query(state: State):
    llm = get_model()
    structured_llm = llm.with_structured_output(Search)
    query = structured_llm.invoke(state["question"])
    return {"query": query}


def retrieve(state: State):
    vector_store = get_vector_db()
    query = state["query"]
    retrieved_docs = vector_store.similarity_search(
        query["query"],
        filter=lambda doc: doc.metadata.get("section") == query["section"],
    )
    return {"context": retrieved_docs}


def generate(state: State):
    llm = get_model()
    prompt = get_prompt()
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def get_graph():
    graph_builder = StateGraph(State).add_sequence([analyze_query, retrieve, generate])
    graph_builder.add_edge(START, "analyze_query")
    graph = graph_builder.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))
    return graph


def chat():
    graph = get_graph()
    for step in graph.stream(
            {"question": "What does the end of the post say about Task Decomposition?"},
            stream_mode="updates",
    ):
        print(f"{step}\n\n----------------\n")

if __name__ == '__main__':
    set_proxy()
    chat()
    # split_phases()