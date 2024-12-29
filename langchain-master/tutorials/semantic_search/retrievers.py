'''
@Project ：langchain-master 
@File    ：retrievers.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/27 17:38 
'''
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import chain

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.globals import set_debug, get_debug

# 设置调试模式
set_debug(True)
# 获取当前的调试模式
debug_mode = get_debug()

def get_docs():
    documents = [
        Document(
            page_content="Dogs are great companions, known for their loyalty and friendliness.",
            metadata={"source": "mammal-pets-doc"},
        ),
        Document(
            page_content="Cats are independent pets that often enjoy their own space.",
            metadata={"source": "mammal-pets-doc"},
        ),
    ]

    file_path = "../datas/nke-10k-2023.pdf"
    loader = PyPDFLoader(file_path)

    docs = loader.load()
    return docs

def split_docs():
    docs = get_docs()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits

def get_chroma():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(embedding_function=embeddings)
    all_splits = split_docs()
    ids = vector_store.add_documents(documents=all_splits)
    return vector_store, ids

@chain
def retriever(query: str) -> List[Document]:
    vector_store, _ = get_chroma()
    return vector_store.similarity_search(query, k=1)


def do_retrieve():
    response = retriever.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ],
    )
    for res in response:
        print(res)
        print("**********************************")
def vector_store_retrieve():
    vector_store, _ = get_chroma()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 1},
    )
    response = retriever.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ],
    )
    for res in response:
        print(res)
        print("-------------------------------")

if __name__ == '__main__':
    if True:
        do_retrieve() # 有问题
    else:
        vector_store_retrieve()