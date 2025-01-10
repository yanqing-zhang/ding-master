'''
@Project ：langchain-master 
@File    ：sql_test.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2025/1/9 9:21 
'''
import os
import getpass
from typing_extensions import TypedDict
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from langgraph.graph import START, StateGraph
from IPython.display import Image, display
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain import hub
from langgraph.prebuilt import create_react_agent
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.agents.agent_toolkits import create_retriever_tool
import ast
import re

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

def get_db():
    # 数据库相关参数设置，包含用户名、密码等
    db_user = "root"
    db_password = "123456"
    db_host = "127.0.0.1:3306"
    db_name = "students"

    # 创建SQLDatabase实例
    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

    return db

def db_test():
    db = get_db()
    print(db.dialect)
    print(db.get_usable_table_names())
    ret = db.run("SELECT * FROM stu_base_info LIMIT 10;")
    print(f"ret:{ret}")

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

def get_model():
    llm = ChatOpenAI(model="gpt-4o-mini")
    return llm

def get_prompt_template():
    query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
    assert len(query_prompt_template.messages) == 1
    query_prompt_template.messages[0].pretty_print()
    return query_prompt_template

class QueryOutput(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    """Generate SQL query to fetch information."""
    query_prompt_template = get_prompt_template()
    db = get_db()
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"],
        }
    )
    llm = get_model()
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

def write_query_test():
    write_query({"question": "How many Employees are there?"})

def execute_query(state: State):
    """Execute SQL query."""
    db = get_db()
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}

def execute_query_test():
    ret = execute_query({"query": "SELECT COUNT(stu_no) AS StuCount FROM stu_base_info;"})
    print(f"ret:{ret}")

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    llm = get_model()
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

def get_graph_builder():
    graph_builder = StateGraph(State).add_sequence(
        [write_query, execute_query, generate_answer]
    )
    graph_builder.add_edge(START, "write_query")
    graph = graph_builder.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))
    return graph_builder, graph

def get_graph_test():
    _, graph = get_graph_builder()
    for step in graph.stream(
        {"question": "How many employees are there?"}, stream_mode="updates"
    ):
        print(step)

def get_memory():
    memory = MemorySaver()
    return memory


def chat():
    graph_builder,_ = get_graph_builder()
    memory = get_memory()
    graph = graph_builder.compile(checkpointer=memory, interrupt_before=["execute_query"])

    # Now that we're using persistence, we need to specify a thread ID
    # so that we can continue the run after review.
    config = {"configurable": {"thread_id": "1"}}


    display(Image(graph.get_graph().draw_mermaid_png()))


    for step in graph.stream(
        {"question": "How many employees are there?"},
        config,
        stream_mode="updates",
    ):
        print(step)

    try:
        user_approval = input("Do you want to go to execute query? (yes/no): ")
    except Exception:
        user_approval = "no"

    if user_approval.lower() == "yes":
        # If approved, continue the graph execution
        for step in graph.stream(None, config, stream_mode="updates"):
            print(step)
    else:
        print("Operation cancelled by user.")



def get_agent_tools():
    db = get_db()
    llm = get_model()
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    return tools


def get_agent_template():
    prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")

    assert len(prompt_template.messages) == 1
    prompt_template.messages[0].pretty_print()

    system_message = prompt_template.format(dialect="mysql", top_k=5)
    return system_message

def get_agent_executor():
    llm = get_model()
    tools = get_agent_tools()
    system_message = get_agent_template()
    agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
    return agent_executor

def run_agent():
    agent_executor = get_agent_executor()
    question = "哪个学生的数学成成绩最好?"

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    question = "哪门课程的平均成绩最高?"

    for step in agent_executor.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()




def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

def query_db():
    db = get_db()
    stus = query_as_list(db, "SELECT stu_name FROM stu_base_info")
    coures = query_as_list(db, "SELECT course_name FROM stu_courses")
    print(f"top 5:{coures[:5]}")
    return stus, coures



def embedding_and_store():
    stus, coures = query_db()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = InMemoryVectorStore(embeddings)
    _ = vector_store.add_texts(stus + coures)
    return vector_store

def retriever_vector_store():
    vector_store = embedding_and_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    description = (
        "Use to look up values to filter on. Input is an approximate spelling "
        "of the proper noun, output is valid proper nouns. Use the noun most "
        "similar to the search."
    )
    retriever_tool = create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description=description,
    )
    print(retriever_tool.invoke("钱芳燕"))
    return retriever_tool

def retrieve_tool_test():
    retriever_tool = retriever_vector_store()
    system_message = get_agent_template()
    llm = get_model()
    tools = get_agent_tools()
    # Add to system message
    suffix = (
        "If you need to filter on a proper noun like a Name, you must ALWAYS first look up "
        "the filter value using the 'search_proper_nouns' tool! Do not try to "
        "guess at the proper name - use this function to find similar ones."
    )

    system = f"{system_message}\n\n{suffix}"

    tools.append(retriever_tool)

    agent = create_react_agent(llm, tools, state_modifier=system)

    question = "钱芳燕的各科成绩分别是多少?"

    for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()



if __name__ == '__main__':
    if True:
        retrieve_tool_test()
    else:
        db_test()
        write_query_test()
        get_graph_test()
        execute_query_test()
        chat()
