'''
@Project ：langchain-master 
@File    ：extraction_chain.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/31 18:37 
'''
import getpass
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from typing import List, Optional

from pydantic import BaseModel, Field

from langchain_core.utils.function_calling import tool_example_to_messages

"""
通过langchain提取链可以把非结构化的文本进行结构化
"""

class Person(BaseModel):
    """Information about a person."""

    # ^ Doc-string for the entity Person.
    # This doc-string is sent to the LLM as the description of the schema Person,
    # and it can help to improve extraction results.

    # Note that:
    # 1. Each field is an `optional` -- this allows the model to decline to extract it!
    # 2. Each field has a `description` -- this description is used by the LLM.
    # Having a good description can help improve extraction results.
    name: Optional[str] = Field(default=None, description="The name of the person")
    hair_color: Optional[str] = Field(
        default=None, description="The color of the person's hair if known"
    )
    height_in_meters: Optional[str] = Field(
        default=None, description="Height measured in meters"
    )


class Data(BaseModel):
    """Extracted data about people."""
    # Creates a model so that we can extract multiple entities.
    people: List[Person]



# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked to extract, "
            "return null for the attribute's value.",
        ),
        # Please see the how-to about improving performance with
        # reference examples.
        # MessagesPlaceholder('examples'),
        ("human", "{text}"),
    ]
)


if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


def get_llm():
    llm = ChatOpenAI(model="gpt-4o-mini")
    return llm


def extract_person():
    llm = get_llm()
    structured_llm = llm.with_structured_output(schema=Person)
    text = "Alan Smith is 6 feet tall and has blond hair."
    prompt = prompt_template.invoke({"text": text})

    response = structured_llm.invoke(prompt)
    print(f"response:{response}")

def extract_data():
    llm = get_llm()
    structured_llm = llm.with_structured_output(schema=Data)
    text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
    prompt = prompt_template.invoke({"text": text})
    response = structured_llm.invoke(prompt)
    print(f"response:{response}")

def extract_example():
    """
    🦜代表的是一种未知的计算符号，一开始也不知道他是+或-或×或÷，
    但通过几个示例，大模型就可以推断出🦜的是哪个计算符号，从而能计算出 3 🦜 4的结果是7
    """
    messages = [
        {"role": "user", "content": "2 🦜 2"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "2 🦜 3"},
        {"role": "assistant", "content": "5"},
        {"role": "user", "content": "3 🦜 4"},
    ]
    llm = get_llm()
    response = llm.invoke(messages)
    print(response.content)



def extract_tool_call_example():
    examples = [
        (
            "The ocean is vast and blue. It's more than 20,000 feet deep.",
            Data(people=[]),
        ),
        (
            "Fiona traveled far from France to Spain.",
            Data(people=[Person(name="Fiona", height_in_meters=None, hair_color=None)]),
        ),
    ]
    messages = []

    for txt, tool_call in examples:
        if tool_call.people:
            # This final message is optional for some providers
            ai_response = "Detected people."
        else:
            ai_response = "Detected no people."
        messages.extend(tool_example_to_messages(txt, [tool_call], ai_response=ai_response))

    for message in messages:
        message.pretty_print()

    extract_tool_call_people(messages)

def extract_tool_call_people(messages):
    llm = get_llm()
    message_no_extraction = {
        "role": "user",
        "content": "The solar system is large, but earth has only 1 moon.",
    }

    structured_llm = llm.with_structured_output(schema=Data)
    response = structured_llm.invoke([message_no_extraction])
    print(f"response:{response}")

    result = structured_llm.invoke(messages + [message_no_extraction])
    print(f"result:{result}")

if __name__ == '__main__':
    if True:
        extract_tool_call_example()
    else:
        extract_person()
        extract_data()
        extract_example()
