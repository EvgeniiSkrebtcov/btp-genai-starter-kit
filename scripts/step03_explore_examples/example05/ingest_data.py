from library.constants.folders import FILE_ENV
from library.util.logging import initLogger

import os
from unstructured.partition.pdf import partition_pdf

from dotenv import load_dotenv
import logging

log = logging.getLogger(__name__)
initLogger()


# This function loads the documents into the HANA DB to get them vectorized and validates the documents are loaded correctly
def main():
    # Load environment variables
    load_dotenv(dotenv_path=str(FILE_ENV), verbose=True)

    input_path = os.getcwd()
    print(input_path)
    output_path = os.path.join(os.getcwd(), "output")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Get elements
    raw_pdf_elements = partition_pdf(
        filename=os.path.join(script_dir, "TDS LOCTITE HY 4090-EN.pdf"),
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=output_path,
    )

    # # Get the proxy client for the AI Core service
    # proxy_client = get_proxy_client("gen-ai-hub")
    # # Create the OpenAIEmbeddings object
    # embeddings = OpenAIEmbeddings(
    #     proxy_model_name="text-embedding-ada-002", proxy_client=proxy_client
    # )

    # # Create the HanaDB object
    # db = HanaDB(
    #     embedding=embeddings,
    #     connection=connection_to_hana,
    #     table_name=VECTOR_EMBEDDINGS_TABLE_NAME,
    # )

    # # Delete already existing documents from the table
    # db.delete(filter={})
    # log.info("Cleaning up table with vectoe embeddings.")

    # # add the loaded document chunks to the HANA DB
    # log.info("Adding the loaded document chunks to the HANA DB ...")
    # db.add_documents(chunks)
    # log.success("Done!")

    # # -------------------------------------------------------------------------------------
    # # Validate the documents are loaded correctly
    # # -------------------------------------------------------------------------------------
    # log.info("Validate the documents are loaded correctly")
    # cur = connection_to_hana.cursor()
    # cur.execute(
    #     f"SELECT VEC_TEXT, VEC_META, TO_NVARCHAR(VEC_VECTOR) FROM {VECTOR_EMBEDDINGS_TABLE_NAME} LIMIT 1"
    # )

    # rows = cur.fetchall()
    # print(rows[0][0])  # The text
    # print(rows[0][1])  # The metadata
    # print(
    #     f"{rows[0][2][:100]}..."
    # )  # The vector (printing only first 100 characters as it is quite long)
    # cur.close()

    # log.success("Data ingestion completed.")


if __name__ == "__main__":
    main()
