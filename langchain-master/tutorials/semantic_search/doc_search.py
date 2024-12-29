'''
@Project ：langchain-master 
@File    ：doc_search.py
@IDE     ：PyCharm 
@Author  ：yanqing.zhang@
@Date    ：2024/12/27 12:38
@reference：https://python.langchain.com/docs/tutorials/retrievers/
'''
import getpass
import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import asyncio

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

    print(f"length of docs:{len(docs)}")
    print(f"{docs[0].page_content[:200]}\n")
    print(f"meta:{docs[0].metadata}")
    return docs

def split_docs():
    docs = get_docs()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    len(all_splits)
    return all_splits

def do_embeddings():
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    all_splits = split_docs()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    vector_2 = embeddings.embed_query(all_splits[1].page_content)

    assert len(vector_1) == len(vector_2)
    print(f"Generated vectors of length {len(vector_1)}\n")
    print(vector_1[:10])

def get_chroma():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vector_store = Chroma(embedding_function=embeddings)
    all_splits = split_docs()
    ids = vector_store.add_documents(documents=all_splits)
    return vector_store, ids

def similarity_search():
    vector_store,_ = get_chroma()
    results = vector_store.similarity_search(
        "How many distribution centers does Nike have in the US?"
    )
    print(f"result:{results[0]}")

async def async_similarity_search():
    vector_store, _ = get_chroma()
    results = await vector_store.asimilarity_search("When was Nike incorporated?")
    print(f"result:{results[0]}")

def get_scores_similarity_search():
    vector_store, _ = get_chroma()
    results = vector_store.similarity_search_with_score("What was Nike's revenue in 2023?")
    doc, score = results[0]
    print(f"Score: {score}\n")
    print(f"doc:{doc}")
    print("------------------------")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embedding = embeddings.embed_query("How were Nike's margins impacted in 2023?")
    results = vector_store.similarity_search_by_vector(embedding)
    print(results[0])


if __name__ == '__main__':
    if True:
        # async_similarity_search()
        get_scores_similarity_search()
    else:
        split_docs()
        do_embeddings()
        similarity_search()