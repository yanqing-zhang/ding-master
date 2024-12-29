'''
@Project ：langchain-master 
@File    ：qa_over_document.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/22 11:02 
'''
import os
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import SimpleMemory
from langchain_community.utilities import SerpAPIWrapper

def web_query_test():
    # 初始化LLM和WebBrowser
    serp_api_key = os.environ['SERP_API_KEY']
    llm = OpenAI(temperature=0.9)
    web_browser = SerpAPIWrapper(serpapi_api_key=serp_api_key)

    # 创建一个记忆组件，用于存储历史信息
    memory = SimpleMemory()

    # 创建一个模板，用于生成搜索查询
    search_template = PromptTemplate(
        input_variables=["search_query"],
        template="Search for information on the webpage {search_query}"
    )

    # 创建一个LLM链，用于处理搜索结果
    search_chain = LLMChain(
        llm=llm,
        prompt=search_template,
        memory=memory
    )

    # 定义要搜索的网页
    webpage_url = "https://lilianweng.github.io/posts/2023-06-23-agent/"

    # 执行搜索
    search_query = "LLM-powered autonomous agents"
    search_results = web_browser.search(search_query, webpage_url)

    # 使用LLM链处理搜索结果
    # 注意：这里使用 invoke 方法替代了原先的 __call__ 方法
    response = search_chain.run(search_query=search_query)

    print("Search Results:", search_results)
    print("LLM Response:", response)



if __name__ == '__main__':
    os.environ['http_proxy'] = 'http://127.0.0.1:10792'
    os.environ['https_proxy'] = 'http://127.0.0.1:10792'
    web_query_test()