'''
@Project ：langchain-master 
@File    ：classifications.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/30 19:38 
'''
import getpass
import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

def get_prompt():
    tagging_prompt = ChatPromptTemplate.from_template(
        """
            从以下段落中提取所需信息。
            
            仅提取 #Classification# 类中提到的属性值
            
            段落如下:
            {input}
        """
    )
    return tagging_prompt


class Classification(BaseModel):
    sentiment: str = Field(description="文本的情感")
    aggressiveness: int = Field(
        description="文本的攻击性程度从1到10"
    )
    language: str = Field(description="文本所使用的语言")


def get_model():
    """
    必须要使用下面的.with_structured_output(Classification)才可以成功提取Classification类的属性值，
    如果只是使用llm = ChatOpenAI(model="gpt-4o-mini")则会失败，不能与Classification进行关联
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
        Classification
    )
    return llm

def invoke_1():
    llm = get_model()
    tagging_prompt = get_prompt()

    # inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
    inp = "第一次自己买电脑，看了好几家品牌，多种类型挑选了戴尔的这一型号。包装严密，外观大气，尺寸小，不占地方，显色清晰，反应迅速。因为不懂，电脑的安装从头到尾，详细咨询客服，客服都耐心地给我明确而快速的解答，非常满意，感谢客服。"

    prompt = tagging_prompt.invoke({"input": inp})
    print(f"prompt:{prompt}")
    print("====================================")
    response = llm.invoke(prompt)
    # content="The passage does not contain any properties mentioned in the 'Classification' function. Therefore, there are no properties to extract."
    print(f"response:{response}")


def invoke_2():
    llm = get_model()
    tagging_prompt = get_prompt()
    inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"

    prompt = tagging_prompt.invoke({"input": inp})
    response = llm.invoke(prompt)

    print(f"response.dict:{response.model_dump()}")

if __name__ == '__main__':
    if True:
        invoke_1()
    else:
        invoke_2()