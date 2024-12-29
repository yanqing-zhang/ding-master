'''
@Project ：langchain-master 
@File    ：chat_model.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/27 11:00
@reference：https://python.langchain.com/docs/tutorials/llm_chain/
'''
import getpass
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

def get_model():
    if not os.environ.get("OPENAI_API_KEY"):
      os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


    model = ChatOpenAI(model="gpt-4o-mini")
    return model

def invoke_message():
    """
    response:content='Ciao!' additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 4, 'prompt_tokens': 20, 'total_tokens': 24, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_d02d531b47', 'finish_reason': 'stop', 'logprobs': None} id='run-c41707a7-407c-4269-a056-f80d596cbb06-0' usage_metadata={'input_tokens': 20, 'output_tokens': 4, 'total_tokens': 24, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
    """
    model = get_model()
    messages = [
        SystemMessage("Translate the following from English into Italian"),
        HumanMessage("hi!"),
    ]
    response = model.invoke(messages)
    print(f"response:{response}")

def multi_invoke_message():
    model = get_model()
    reponse_01 = model.invoke("Hello")
    print(f"reponse_01:{reponse_01}")
    print("----")
    reponse_02 = model.invoke([{"role": "user", "content": "Hello"}])
    print(f"reponse_02:{reponse_02}")
    print("#####")
    reponse_03 = model.invoke([HumanMessage("Hello")])
    print(f"reponse_03:{reponse_03}")
    print("********")

def get_model_stream():
    model = get_model()
    messages = [
        SystemMessage("Translate the following from English into Italian"),
        HumanMessage("hi!"),
    ]
    for token in model.stream(messages):
        print(token.content, end="|")


if __name__ == '__main__':
    if True:
        get_model_stream()
    else:
        invoke_message()
        multi_invoke_message()

