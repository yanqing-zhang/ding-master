'''
@Project ：langchain-master
@File    ：prompts.py
@IDE     ：PyCharm
@Author  ：yanqing.zhang@
@Date    ：2024/11/25 21:42
'''
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
import os
from function_explainer_prompt_template import FunctionExplainerPromptTemplate,test_add

def template_base():
    """
    最简单的提示词模板，挖空儿型
    """
    prompt_template = PromptTemplate.from_template("请给我讲一个关于{content}的{adjective}的笑话")
    prompt = prompt_template.format(content="小鸡", adjective="有意思")
    print(f"prompt:{prompt}")

def chat_template():
    """
    三种角色对话模板:system、human、ai
    """
    template = ChatPromptTemplate.from_messages([
        ("system", "你是一位得力的AI机器人。你的名字是{user_name}"),
        ("human", "你好，你在做什么？"),
        ("ai", "我很好，谢谢!"),
        ("human", "{user_input}"),
    ])
    messages = template.format_messages(
        user_name="张三",
        user_input="你叫什么名字?"
    )

    print(f"message:{messages}")

def message_template():
    """
    两种角色对话模板:SystemMessage、HumanMessage
    """
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=("你是一位乐于助人的助手，能够重新编写用户的 text "
                         "使其听起来更加积极向上。"
                         )
            ),
            HumanMessagePromptTemplate.from_template("{text}")
        ]
    )

    GPT_MODEL = "gpt-3.5-turbo"
    openai_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(model=GPT_MODEL,
                       temperature=0.5,
                       timeout=60,
                       max_tokens=1000,
                       max_retries=6,
                       api_key=openai_key)
    response = model.invoke(template.format_messages(text="我不喜欢吃美味的东西。"))
    print(f"response:{response.content}")

def customer_template():
    fn_explainer = FunctionExplainerPromptTemplate(input_variables=["function_name"])
    prompt_1 = fn_explainer.format(function_name=test_add)
    print(f"prompt_1:{prompt_1}")
    chat_prompt = ChatPromptTemplate.from_messages([prompt_1])
    GPT_MODEL = "gpt-3.5-turbo"
    openai_key = os.getenv("OPENAI_API_KEY")

    model = ChatOpenAI(model=GPT_MODEL,
                       temperature=0.5,
                       timeout=60,
                       max_tokens=1000,
                       max_retries=6,
                       api_key=openai_key)
    response = model.invoke(chat_prompt)
    print(f"response:{response.content}")


if  __name__ == '__main__':
    os.environ['http_proxy'] = 'http://127.0.0.1:10792'
    os.environ['https_proxy'] = 'http://127.0.0.1:10792'
    if True:
        customer_template()
    else:
        template_base()
        chat_template()
        message_template()

