from hr_assistant.logger import get_logger
from hr_assistant import config 
logger = get_logger(__name__)

def check_langsmith_tracing()->None:
    tracing_on =  config.LANGSMITH_TRACING.lower() == "true"

    if tracing_on and config.LANGSMITH_API_KEY:
        logger.info(
            "Langsmith tracing ENABLED -project '%s',traces at %s",
            config.LANGSMITH_PROJECT,
            "https://smith.langchain.com"
        )
    else:
        logger.info("Langsmith tracing is OFF (set LANGSMITH_TRACING=true in .env enable it)")