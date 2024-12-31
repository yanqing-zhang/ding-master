'''
@Project ：langchain-master 
@File    ：finer_control.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/31 17:55 
'''
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
class Classification(BaseModel):
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["spanish", "english", "french", "german", "italian"]
    )

def get_template():
    tagging_prompt = ChatPromptTemplate.from_template(
        """
    Extract the desired information from the following passage.

    Only extract the properties mentioned in the 'Classification' function.

    Passage:
    {input}
    """
    )
    return tagging_prompt
def get_llm():
    """
    必须要使用下面的.with_structured_output(Classification)才可以成功提取Classification类的属性值，
    如果只是使用llm = ChatOpenAI(model="gpt-4o-mini")则会失败，不能与Classification进行关联
    """
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini").with_structured_output(
        Classification
    )
    return llm

def classify_test():
    llm = get_llm()
    tagging_prompt = get_template()
    inp = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
    prompt = tagging_prompt.invoke({"input": inp})
    result1 = llm.invoke(prompt)
    print(f"result1:{result1}")

    inp = "Estoy muy enojado con vos! Te voy a dar tu merecido!"
    prompt = tagging_prompt.invoke({"input": inp})
    result2 = llm.invoke(prompt)
    print(f"result2:{result2}")

    inp = "Weather is ok here, I can go outside without much more than a coat"
    prompt = tagging_prompt.invoke({"input": inp})
    result3 = llm.invoke(prompt)
    print(f"result3:{result3}")

    inp = "第一次自己买电脑，看了好几家品牌，多种类型挑选了戴尔的这一型号。包装严密，外观大气，尺寸小，不占地方，显色清晰，反应迅速。因为不懂，电脑的安装从头到尾，详细咨询客服，客服都耐心地给我明确而快速的解答，非常满意，感谢客服。"
    prompt = tagging_prompt.invoke({"input": inp})
    result4 = llm.invoke(prompt)
    print(f"result4:{result4}")

if __name__ == '__main__':
    classify_test()