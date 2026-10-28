from langchain_groq import ChatGroq
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    logger.info("Intialzing LLM")
    return ChatGroq(model=config.LLM_MODEL_NAME,temperature=0)

PRIMARY_PROVIDER={"provider":"@groq1",
                 "override_params":{"model":config.LLM_MODEL_NAME}}
FALLBACK_PROVIDER={"provider":"@groq2",
                 "override_params":{"model":"openai/gpt-oss-20b"}}


"""GATEWAY_CONFIG={
  "strategy": {
    "mode": "fallback"
  },
  "targets": [
    {
      "provider": "@groq1",
      "override_params": {
        "model": "openai/gpt-oss-20b"
      }
    },
    {
      "provider": "@groq2",
      "override_params": {
        "model": "openai/gpt-oss-20b"
      }
    }
  ]
}"""

def get_gateway_llm()->ChatOpenAI:
    logger.info("Routing LLM Calls through Port Key")
    headers = createHeaders(api_key=config.PORTKEY_API_KEY,config=config.PORTKEY_CONFIG_ID)
    return ChatOpenAI(
        api_key="portkey",
        base_url=PORTKEY_GATEWAY_URL,
        model=config.LLM_MODEL_NAME,
        default_headers=headers,
    )