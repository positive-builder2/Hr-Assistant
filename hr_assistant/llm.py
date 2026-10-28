from langchain_groq import ChatGroq
from hr_assistant import config
from hr_assistant.gateway import get_gateway_llm
from hr_assistant.logger import get_logger

logger = get_logger(__name__)


def get_llm():
    #logger.info("Intializing LLM '%s'",config.LLM_MODEL_NAME)
    #return ChatGroq(model=config.LLM_MODEL_NAME,temperature=0)
    logger.info("Intializing LLM via PortKey")
    return get_gateway_llm()