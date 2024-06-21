import logging

from dotenv import load_dotenv
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores.hanavector import HanaDB
from library.constants.folders import FILE_ENV
from library.constants.table_names import TABLE_NAME
from library.data.hana_db import get_connection_to_hana_db
from library.util.logging import initLogger

log = logging.getLogger(__name__)
initLogger()


# This function loads the documents into the HANA DB to get them vectorized and validates the documents are loaded correctly
def main():
    # Load environment variables
    load_dotenv(dotenv_path=str(FILE_ENV), verbose=True)

    log.header("Load the documents into the HANA DB to get them vectorized")

    # Download transcript from Episode 64 of the SAP podcast
    loader_ep64 = PyMuPDFLoader(
        "https://sap-podcast-bucket.s3.amazonaws.com/the-future-of-supply-chain/The_Future_of_Supply_Chain_Episode_64_transcript.pdf"
    )
    # Download transcript from Episode 65 of the SAP podcast
    loader_ep65 = PyMuPDFLoader(
        "https://sap-podcast-bucket.s3.amazonaws.com/the-future-of-supply-chain/The_Future_of_Supply_Chain_Episode_65_transcript.pdf"
    )

    # Load the documents and split them into chunks
    chunks = loader_ep64.load_and_split()
    chunks += loader_ep65.load_and_split()

    print(chunks[0])

    # Get the connection to the HANA DB
    connection_to_hana = get_connection_to_hana_db()
    log.info("Connection to HANA DB established")

    # Get the proxy client for the AI Core service
    proxy_client = get_proxy_client("gen-ai-hub")
    
    # Create the OpenAIEmbeddings object
    embeddings = OpenAIEmbeddings(
        proxy_model_name="text-embedding-ada-002", proxy_client=proxy_client
    )

    # Create the HanaDB object
    db = HanaDB(
        embedding=embeddings, connection=connection_to_hana, table_name=TABLE_NAME
    )

    # Delete already existing documents from the table
    db.delete(filter={})
    log.info("Deleted already existing documents from the table")

    # add the loaded document chunks to the HANA DB
    log.info("Adding the loaded document chunks to the HANA DB ...")
    db.add_documents(chunks)
    log.success("Done!")


if __name__ == "__main__":
    main()
