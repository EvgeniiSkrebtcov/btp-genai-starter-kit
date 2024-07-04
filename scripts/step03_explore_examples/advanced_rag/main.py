import logging
import sys

from utils.env import init_env
from utils.hana import teardown_hana_table, has_embeddings, check_if_exists

from library.config import TABLE_NAME
from library.loaders import fetch_document_from_git, load_from_dir
from library.ingest import ingest_from_pdf, ingest_from_github
from library.splitter import (
    recursive_split_docs_into_chunks,
    markdown_split_docs_into_chunks,
    semantic_split_docs_into_chunks,
)

log = logging.getLogger(__name__)


def split_documents():
    docs = fetch_document_from_git()
    recursive_split_docs_into_chunks(docs, preview=True)
    markdown_chunks = markdown_split_docs_into_chunks(docs, preview=True)
    recursive_split_docs_into_chunks(markdown_chunks, preview=True)

    # Load the documents from a directory
    docs_from_dir = load_from_dir(input_dir="./gen/docs/assets")
    semantic_split_docs_into_chunks(docs_from_dir, preview=True)


def exec_sample_query_sample():
    ingest_from_pdf(
        urls=[
            "https://sap-podcast-bucket.s3.amazonaws.com/the-future-of-supply-chain/The_Future_of_Supply_Chain_Episode_64_transcript.pdf",
            "https://sap-podcast-bucket.s3.amazonaws.com/the-future-of-supply-chain/The_Future_of_Supply_Chain_Episode_65_transcript.pdf",
        ]
    )


def main():
    # Load environment variables
    init_env()

    # -------------------------------------------------------------------------------------
    # Provide the response to the user
    # -------------------------------------------------------------------------------------

    print("Welcome to the interactive Q&A session\n")

    while True:
        print("0: Clean up database")
        print("1: Compare Splitter Methods")
        print("2: Advanced RAG - Self Query")
        print("3: Advanced RAG - Rewrite Retrieve Read")
        print("4: Advanced RAG - RAG Fusion")
        print("5: Exit\n")

        option = input("Which task would you like to run?").strip()

        if option == "0":
            teardown_hana_table(TABLE_NAME)
            continue
        elif option == "1":
            split_documents()
            continue
        elif option == "2":
            exec_sample_query_sample()
            break
        elif option == "5":
            print("Goodbye!")
            sys.exit()
        else:
            print("Invalid input. Please choose an option from the list above.")


if __name__ == "__main__":
    main()
