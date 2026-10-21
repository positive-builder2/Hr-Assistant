from hr_assistant import config
from hr_assistant.agent import create_hr_agent
from hr_assistant.document_loader import load_document
from hr_assistant.llm import get_llm
from hr_assistant.splitter import split_into_chunks
from hr_assistant.tools import create_search_tool

from hr_assistant.vector_store import (
    build_vector_store,
    get_retriever,
    load_vector_store,
    save_vector_store,
    vector_store_exists,
)

def build_vector_store_for_document(file_path:str=config.DATA_FILE_PATH):
    if vector_store_exists():
        print("Found a saved Vector Store on Disk,loading it (fast no re-embeddings)")
        return load_vector_store()
    print("No Vector Store found,building one from scratch...")
    documents=load_document(file_path)
    chunks=split_into_chunks(documents)
    print(f"Loaded `{file_path}` and split it into {len(chunks)} chunks")
    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    print("Vector store built and saved to disk for next time")
    return vector_store

def build_hr_assistant(file_path:str=config.DATA_FILE_PATH):
    config.check_api_keys()
    vector_store=build_vector_store_for_document(file_path)
    retriver=get_retriever(vector_store)
    search_tool = create_search_tool(retriver)

    llm = get_llm()
    agent = create_hr_agent(llm,[search_tool])
    return agent

def ask(agent,question:str)->str:
    response = agent.invoke({"messages":[{"role":"user","content":question}]})
    return response["messages"][-1].content
