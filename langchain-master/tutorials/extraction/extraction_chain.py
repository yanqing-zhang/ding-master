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


def get_structured_llm():
    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(schema=Person)

def extract_person():
    structured_llm = get_structured_llm()
    text = "Alan Smith is 6 feet tall and has blond hair."
    prompt = prompt_template.invoke({"text": text})

    response = structured_llm.invoke(prompt)
    print(f"response:{response}")

def extract_data():
    structured_llm = llm.with_structured_output(schema=Data)
    text = "My name is Jeff, my hair is black and i am 6 feet tall. Anna has the same color hair as me."
    prompt = prompt_template.invoke({"text": text})
    structured_llm.invoke(prompt)


messages = [
    {"role": "user", "content": "2 🦜 2"},
    {"role": "assistant", "content": "4"},
    {"role": "user", "content": "2 🦜 3"},
    {"role": "assistant", "content": "5"},
    {"role": "user", "content": "3 🦜 4"},
]

response = llm.invoke(messages)
print(response.content)