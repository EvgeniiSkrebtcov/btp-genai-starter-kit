import logging
import re

from dotenv import load_dotenv
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from langchain_community.document_loaders import GitLoader
from langchain_community.vectorstores.hanavector import HanaDB
from library.constants.folders import FILE_ENV
from library.constants.table_names import TABLE_NAME
from library.data.data_store import split_docs_into_chunks
from library.data.hana_db import get_connection_to_hana_db
from library.util.logging import initLogger

log = logging.getLogger(__name__)
initLogger()


# This function loads the documents into the HANA DB to get them vectorized and validates the documents are loaded correctly
def main():
    # Load environment variables
    load_dotenv(dotenv_path=str(FILE_ENV), verbose=True)

    # Load the documents from a GitHub repository
    log.header("Load the documents into the HANA DB to get them vectorized")
    chunks = load_documents_from_github()

    # Setup HANA DB connection
    log.info("Setup HANA DB connection")
    db = setup_hana_db()

    # Ingest data
    log.info("Ingest chunks into database")
    ingest_data(db, chunks)

    log.success("Data ingestion completed.")


def load_documents_from_github():
    # Load the documents from a GitHub repository
    loader = GitLoader(
        clone_url="https://github.com/SAP-docs/btp-cloud-platform",
        repo_path="./gen/btp-cloud-platform",
        file_filter=lambda file_path: re.match(
            r"^./gen/btp-cloud-platform/docs/10-concepts/.*.md$", file_path
        ),
        branch="main",
    )
    text_documents = loader.load()
    log.info("Getting the documents from the GitHub repository ...")

    # Split the documents into chunks
    chunks = split_docs_into_chunks(documents=text_documents)
    return chunks


def setup_hana_db():
    # Get the connection to the HANA DB
    connection_to_hana = get_connection_to_hana_db()
    log.info("Connection to HANA DB established")

    # Get the proxy client for the AI Core service
    proxy_client = get_proxy_client("gen-ai-hub")
    # Create the OpenAIEmbeddings object
    embeddings = OpenAIEmbeddings(
        proxy_model_name="text-embedding-ada-002", proxy_client=proxy_client
    )

    # Create the HANA DB object
    db = HanaDB(
        embedding=embeddings, connection=connection_to_hana, table_name=TABLE_NAME
    )

    return db


def ingest_data(db, chunks):
    # Delete already existing documents from the table
    db.delete(filter={})
    log.info("Deleted already existing documents from the table")

    # add the loaded document chunks to the HANA DB
    log.info("Adding the loaded document chunks to the HANA DB ...")
    db.add_documents(chunks)


if __name__ == "__main__":
    main()
