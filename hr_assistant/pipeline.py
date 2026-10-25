from hr_assistant import config
from hr_assistant.agent import create_hr_agent
from hr_assistant.document_loader import load_document
from hr_assistant.llm import get_llm
from hr_assistant.splitter import split_into_chunks
from hr_assistant.tools import create_search_tool

from hr_assistant.logger import get_logger
from hr_assistant.tracing import check_langsmith_tracing
from hr_assistant.guardrails import check_input,check_output,REFUSAL_MESSAGE
from hr_assistant.vector_store import (
    build_vector_store,
    get_retriever,
    load_vector_store,
    save_vector_store,
    vector_store_exists,
)

logger = get_logger(__name__)

def build_vector_store_for_document(file_path:str=config.DATA_FILE_PATH):
    if vector_store_exists():
        print("Found a saved Vector Store on Disk,loading it (fast no re-embeddings)")
        logger.info("Found a saved Vector Store on Disk,loading it (fast no re-embeddings)")
        return load_vector_store()
    print("No Vector Store found,building one from scratch...")
    logger.info("No Vector Store found,building one from scratch...")
    documents=load_document(file_path)
    chunks=split_into_chunks(documents)
    print(f"Loaded `{file_path}` and split it into {len(chunks)} chunks")
    #logger.info(f"Loaded `{file_path}` and split it into {len(chunks)} chunks")
    vector_store = build_vector_store(chunks)
    save_vector_store(vector_store)
    logger.info("Vector store built and saved to disk for next time")
    print("Vector store built and saved to disk for next time")
    return vector_store

def build_hr_assistant(file_path:str=config.DATA_FILE_PATH):
    config.check_api_keys()
    check_langsmith_tracing()
    logger.info("Building HR assistant")
    vector_store=build_vector_store_for_document(file_path)
    retriver=get_retriever(vector_store)
    search_tool = create_search_tool(retriver)

    llm = get_llm()
    agent = create_hr_agent(llm,[search_tool])
    logger.info("HR assistant is ready to take questions")
    return agent

def ask(agent,question:str)->str:
    logger.info("User Question: %s",question)

    input_is_safe,_ = check_input(question)
    if not input_is_safe:
        return REFUSAL_MESSAGE
    response = agent.invoke({"messages":[{"role":"user","content":question}]})
    answer = response["messages"][-1].content
    logger.info("Final answer: %s",answer)
    ouput_is_safe,_ = check_output(answer)
    if not ouput_is_safe:
        return REFUSAL_MESSAGE
    return answer
