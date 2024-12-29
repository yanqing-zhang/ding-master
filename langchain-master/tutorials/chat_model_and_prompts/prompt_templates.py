'''
@Project ：langchain-master 
@File    ：prompt_templates.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/27 12:27
@reference：https://python.langchain.com/docs/tutorials/llm_chain/
'''
from langchain_core.prompts import ChatPromptTemplate

def get_prompt():
    """
    prompt:messages=[SystemMessage(content='Translate the following from English into Italian', additional_kwargs={}, response_metadata={}), HumanMessage(content='hi!', additional_kwargs={}, response_metadata={})]
    """
    system_template = "Translate the following from English into {language}"

    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )

    prompt = prompt_template.invoke({"language": "Italian", "text": "hi!"})

    print(f"prompt:{prompt}")


if __name__ == '__main__':
    get_prompt()