import logging

from langchain_community.vectorstores.hanavector import HanaDB
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

from utils.hana import get_connection_to_hana_db

from .config import LLM_MODEL_NAME, EMBEDDINGS_MODEL_NAME, TABLE_NAME

log = logging.getLogger(__name__)


def create_llm_and_embeddings():
    # Get the proxy client for the AI Core service
    proxy_client = get_proxy_client("gen-ai-hub")

    llm = ChatOpenAI(
        proxy_model_name=LLM_MODEL_NAME, proxy_client=proxy_client, temperature=0
    )
    embeddings = OpenAIEmbeddings(
        proxy_model_name=EMBEDDINGS_MODEL_NAME, proxy_client=proxy_client
    )
    return llm, embeddings


def create_hana_connection(**options):
    _, embeddings = create_llm_and_embeddings()
    connection = get_connection_to_hana_db()

    db = HanaDB(embedding=embeddings, connection=connection, table_name=TABLE_NAME)

    if options["cleanup"] is True:
        log.info("Deleting already existing documents from the table")
        db.delete(filter={})
        log.success("Deleted already existing documents from the table")

    return db
