import sys
import logging

from .loaders import load_and_split_pdf
from .factory import create_hana_connection

log = logging.getLogger(__name__)


def ingest_from_pdf(urls: list[str], **options):
    # TODO: Clarify if ingestion should be skipped if the table already has embeddings
    # table_exists = check_if_exists(TABLE_NAME)

    cleanup = options.get("cleanup", False)

    try:
        log.info("Starting ingestion...")

        chunks = list()
        for url in urls:
            chunks += load_and_split_pdf(url)
        db = create_hana_connection(cleanup=cleanup)

        db.add_documents(chunks)
        log.success("Ingestion completed successfully")
    except Exception as e:
        log.error(f"An error occurred during ingestion: {str(e)}")
        sys.exit()


def ingest_from_github():
    pass
