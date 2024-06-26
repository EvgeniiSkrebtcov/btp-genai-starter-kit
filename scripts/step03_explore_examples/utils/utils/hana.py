from logging import getLogger
import os
import sys
from hdbcli import dbapi
from .env import assert_env
from .logging import initLogger

log = getLogger(__name__)
initLogger()


def teardown_hana_table(table_name):
    # Initialize the vector store and storage layer
    try:
        connection_to_hana = get_connection_to_hana_db()
        cur = connection_to_hana.cursor()
        cur.execute(f"DROP TABLE {table_name}")
        cur.close()
    except Exception as e:
        log.error(f"Error dropping table: {str(e)}")


# Function to get a connection to the HANA DB
def get_connection_to_hana_db():
    try:
        assert_env(
            [
                "HANA_DB_ADDRESS",
                "HANA_DB_PORT",
                "HANA_DB_USER",
                "HANA_DB_PASSWORD",
            ]
        )
        conn = dbapi.connect(
            address=os.environ.get("HANA_DB_ADDRESS"),
            port=os.environ.get("HANA_DB_PORT"),
            user=os.environ.get("HANA_DB_USER"),
            password=os.environ.get("HANA_DB_PASSWORD"),
            encrypt=True,
            sslValidateCertificate=False,
        )
        return conn
    except Exception as e:
        log.error(f"Error connecting to HANA DB: {str(e)}")
        sys.exit()


def get_connection_string():
    return "hana+hdbcli://{user}:{password}@{address}:{port}?encrypt=true".format(
        user=os.environ.get("HANA_DB_USER"),
        password=os.environ.get("HANA_DB_PASSWORD"),
        address=os.environ.get("HANA_DB_ADDRESS"),
        port=os.environ.get("HANA_DB_PORT"),
    )
