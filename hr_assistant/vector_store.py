import os

from hr_assistant import config
from langchain_community.vectorstores import FAISS
from hr_assistant.embeddings import get_embeddings_model

def build_vector_store(chunks):
    embeddings_models = get_embeddings_model()
    return FAISS.from_documents(chunks,embeddings_models)

def save_vector_store(vector_store,path:str = config.VECTOR_STORE_PATH):
    vector_store.save_local(path)

def load_vector_store(path:str = config.VECTOR_STORE_PATH):
    embeddings_model=get_embeddings_model()
    return FAISS.load_local(path,embeddings_model,allow_dangerous_deserialization=True)

def vector_store_exists(path:str = config.VECTOR_STORE_PATH)->bool:
    return os.path.exists(os.path.join(path,'index.faiss'))

def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    return vector_store.as_retriever(search_kwargs={"k": k})