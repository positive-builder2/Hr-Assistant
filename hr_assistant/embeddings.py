from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config

def get_embeddings_model():
    return JinaEmbeddings(model=config.EMBEDDING_MODEL_NAME)
