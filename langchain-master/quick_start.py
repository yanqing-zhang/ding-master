'''
@Project ：langchain-master 
@File    ：quick_start.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/11/14 21:42 
'''
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain.chains.llm import LLMChain
from pydantic import Field,BaseModel

class GetWeather(BaseModel):
    """Get the current weather in a given location"""
    location: str = Field(..., description="城市, 例如: 北京, 上海")

class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""
    def parse(self, text: str):
        """Parse the output of an LLM call."""
        return text.strip().split(",")

GPT_MODEL = "gpt-3.5-turbo"
openai_key = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(model= GPT_MODEL,
                   temperature=0.5,
                   timeout=60,
                   max_tokens=1000,
                   max_retries=6,
                   api_key=openai_key)

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

system_template = "Translate the following from English into {language}"

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

def chat_base():
    messages = [
        (
            "system",
            "你是一位得力的翻译助理，擅长把中文翻译成英文。要翻译的句子如下：",
        ),
        ("human", "我喜欢徐州宝莲寺"),
    ]
    result = model.invoke(messages)
    print(f"result: {result.content}")

def chain_base():
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一位得力的翻译助理，擅长把 {input_language} 翻译成 {output_language} 。",
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | model
    response = chain.invoke(
        {
            "input_language": "中文",
            "output_language": "Englishi",
            "input": "徐州市的宝莲寺真是一座漂亮的旅游景点。",
        }
    )
    print(f"response: {response.content}")

def tools_base():
    llm_with_tools  = model.bind_tools([GetWeather])
    response_msg = llm_with_tools.invoke(
        "北京今天是什么天气",
    )
    print(f"response_msg:{response_msg}")

def templates_base():
    template = "你是一位得力的翻译助理，擅长把 {input_language} 翻译成 {output_language} 。"
    system_template = SystemMessagePromptTemplate.from_template(template)
    human_text = "{text}"
    human_template = HumanMessagePromptTemplate.from_template(human_text) # from_template
    chat_template = ChatPromptTemplate.from_messages([system_template,human_template]) # from_messages
    print(f"chat_template:{chat_template}")
    chat_prompt = chat_template.format_messages(input_language="中文",
                                              output_language="英文",
                                              text="徐州市的宝莲寺真是一座漂亮的旅游景点。") # format_messages
    response = model.invoke(chat_prompt)
    print(f"response:{response.content}")

def output_parser_base():
    parser = CommaSeparatedListOutputParser()
    messages = "徐州市的宝莲寺,真是一座漂亮的旅游景点。"
    result = parser.parse(messages)
    print(f"result:{result}")

def chain_2():
    """
    老旧用法被警告：
    LangChainDeprecationWarning: The class `LLMChain` was deprecated in LangChain 0.1.17 and will be removed in 1.0. Use :meth:`~RunnableSequence,
    e.g., `prompt | llm`` instead.
    chain = LLMChain(llm=model, prompt=prompt)
    见chain_base()用法
    :return:
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一位得力的翻译助理，擅长把 {input_language} 翻译成 {output_language} 。",
            ),
            ("human", "{input}"),
        ]
    )
    prompt_template = "请把下面的 {input_language} 翻译成 {output_language}，文章如下{text}"
    chain = LLMChain(llm=model, prompt=prompt)
    response = chain.invoke({
            "input_language": "中文",
            "output_language": "Englishi",
            "input": "哈啰单车表示，因系统全面升级，哈啰单车将于2024年11月15日起在郑州市暂停运营，恢复时间另行通知。用户如有需求，可进行骑行卡退款。",
        })
    print(f"response:{response}")


if __name__ == '__main__':
    os.environ['http_proxy'] = 'http://127.0.0.1:10809'
    os.environ['https_proxy'] = 'http://127.0.0.1:10809'
    if False:
        chat_base()
        chain_base()
        tools_base()
        templates_base()
        output_parser_base()
    else:
        chain_2()
    # model.invoke(messages)
    # result = prompt_template.invoke({"language": "Italian", "text": "hi!"})
    # print(f"result: {result}")

