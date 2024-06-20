from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from langchain_community.vectorstores.hanavector import HanaDB
from library.constants.folders import FILE_ENV
from library.constants.table_names import STRUCTURED_DATA_TABLE_NAME, VECTOR_EMBEDDINGS_TABLE_NAME
from library.data.data_store import split_docs_into_chunks
from library.data.hana_db import get_connection_to_hana_db
from library.util.logging import initLogger
from langchain_community.document_loaders import WikipediaLoader
from dotenv import load_dotenv
import logging

log = logging.getLogger(__name__)
initLogger()


# This function loads the documents into the HANA DB to get them vectorized and validates the documents are loaded correctly
def main():
    # Load environment variables
    load_dotenv(dotenv_path=str(FILE_ENV), verbose=True)

    rows = [
        {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
        {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
        {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
    ]

    connection_to_hana = get_connection_to_hana_db()
    log.info("Connection to HANA DB established.")

    cur = connection_to_hana.cursor()

    # # -------------------------------------------------------------------------------------
    # # Create the structured data table
    # # -------------------------------------------------------------------------------------
    log.info("Creating the structured data table...")
    cur.execute(f"DROP TABLE {STRUCTURED_DATA_TABLE_NAME}")
    cur.close()

    cur.execute(f"CREATE TABLE {STRUCTURED_DATA_TABLE_NAME} (CITY_NAME NCHAR(16) PRIMARY KEY, POPULATION INTEGER, COUNTRY NCHAR(16))")
    cur.close()

    sql = f'INSERT INTO {STRUCTURED_DATA_TABLE_NAME} (CITY_NAME, POPULATION, COUNTRY) VALUES (:city_name, :population, :country)'
    for row in rows:
        cur.execute(sql, {"city_name": row["city_name"], "population": row["population"], "country": row["country"]})
    cur.close()
    
    log.info("Table with structured data created:")
    cur.execute(f"SELECT * FROM {STRUCTURED_DATA_TABLE_NAME}")
    print(cur.fetchall())
    cur.close()

    # -------------------------------------------------------------------------------------
    # Creating the vector embeddings table
    # -------------------------------------------------------------------------------------
    # Fetch Wikipedia data for the specified cities
    log.info("Fetching city data from Wikipedia...")
    wiki_docs = [WikipediaLoader(query=row["city_name"], load_max_docs=1).load()[0] for row in rows] 
    log.info(f"Found {len(wiki_docs)} documents from Wikipedia.")

    # Split the documents into chunks
    chunks = split_docs_into_chunks(documents=wiki_docs)

    # Get the proxy client for the AI Core service
    proxy_client = get_proxy_client("gen-ai-hub")
    # Create the OpenAIEmbeddings object
    embeddings = OpenAIEmbeddings(
        proxy_model_name="text-embedding-ada-002", proxy_client=proxy_client
    )

    # Create the HanaDB object
    db = HanaDB(
        embedding=embeddings, connection=connection_to_hana, table_name=VECTOR_EMBEDDINGS_TABLE_NAME
    )

    # Delete already existing documents from the table
    db.delete(filter={})
    log.info("Cleaning up table with vectoe embeddings.")

    # add the loaded document chunks to the HANA DB
    log.info("Adding the loaded document chunks to the HANA DB ...")
    db.add_documents(chunks)
    log.success("Done!")

    # -------------------------------------------------------------------------------------
    # Validate the documents are loaded correctly
    # -------------------------------------------------------------------------------------
    log.info("Validate the documents are loaded correctly")
    cur = connection_to_hana.cursor()
    cur.execute(f"SELECT VEC_TEXT, VEC_META, TO_NVARCHAR(VEC_VECTOR) FROM {VECTOR_EMBEDDINGS_TABLE_NAME} LIMIT 1")

    rows = cur.fetchall()
    print(rows[0][0])  # The text
    print(rows[0][1])  # The metadata
    print(f"{rows[0][2][:100]}...")  # The vector (printing only first 100 characters as it is quite long)
    cur.close()

    log.success("Data ingestion completed.")


if __name__ == "__main__":
    main()
