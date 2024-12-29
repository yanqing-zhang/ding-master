'''
@Project ：langchain-master 
@File    ：quick_start_01.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/23 13:23
@reference : https://python.langchain.com/docs/tutorials/llm_chain/
'''
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.agent_toolkits.load_tools import load_tools

def simple_llm_template_test():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables = ["food"],
        template = "给爱吃{food}的人，推荐5个度假圣地。"
    )
    template = prompt.format(food="甜点")
    response = llm.invoke(template)
    print(f"response:{response}")

def simple_chatllm_template_test():
    GPT_MODEL = "gpt-3.5-turbo"
    model = ChatOpenAI(model=GPT_MODEL)
    messages = [
        SystemMessage("请把下面的文字从中文翻译成英文"),
        HumanMessage("今天（12月23日）早上7点，记者在轨交11号线嘉定北站看到，不断有乘客通过闸机进入站台候车。"),
    ]
    response = model.invoke(messages)
    print(f"response:{response}")
    print("-----------------------------")
    print(f"response:{response.content}")

def simple_chatllm_stream_test():
    """本函数关注的是模型返回内容可以通过流式结构进行显示"""
    GPT_MODEL = "gpt-3.5-turbo"
    model = ChatOpenAI(model=GPT_MODEL)
    messages = [
        SystemMessage("请把下面的文字从中文翻译成英文"),
        HumanMessage("今天（12月23日）早上7点，记者在轨交11号线嘉定北站看到，不断有乘客通过闸机进入站台候车。"),
    ]
    message_chunks = model.stream(messages)
    for token in message_chunks:
        print(token.content, end="|")

def chain_llm_test():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["food"],
        template="给爱吃{food}的人，推荐5个度假圣地。"
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"food": "水果"})
    print(f"response:{response}")

def google_search_agent_test():
    llm = OpenAI(temperature=0.9)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)

if __name__ == '__main__':
    if True:
        chain_llm_test()
    else:
        simple_llm_template_test()
        simple_chatllm_template_test()
        simple_chatllm_stream_test()