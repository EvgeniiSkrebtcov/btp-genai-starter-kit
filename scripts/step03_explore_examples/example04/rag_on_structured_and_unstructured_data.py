from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from library.data.hana_db import get_connection_to_hana_db
from library.constants.table_names import VECTOR_EMBEDDINGS_TABLE_NAME
from library.constants.folders import FILE_ENV
from library.constants.prompts import SQL_AGENT_PREFIX
from library.util.logging import initLogger
from library.data.hana_db import get_connection_string
from library.agents.tools import RAGTool
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.agents.types import AgentType
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.vectorstores.hanavector import HanaDB
from dotenv import load_dotenv
import logging

log = logging.getLogger(__name__)
initLogger()


def main():
    # Load environment variables
    load_dotenv(dotenv_path=str(FILE_ENV), verbose=True)

    # Get the proxy client for the AI Core service
    proxy_client = get_proxy_client("gen-ai-hub")

    # Step 1. Define LLM and embeddings model
    llm = ChatOpenAI(proxy_model_name="gpt-4", proxy_client=proxy_client, temperature=0)
    log.info("ChatOpenAI object created")
    embeddings = OpenAIEmbeddings(
        proxy_model_name="text-embedding-ada-002", proxy_client=proxy_client
    )
    log.info("Embeddings model object created")

    # Step 2. Define the database and the retriever
    connection_to_hana = get_connection_to_hana_db()
    vector_db = HanaDB(
        embedding=embeddings,
        connection=connection_to_hana,
        table_name=VECTOR_EMBEDDINGS_TABLE_NAME,
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 10})

    # Step 3. Create the RAG tool and the SQL agent
    rag_tool = RAGTool(llm=llm, retriever=retriever)
    db = SQLDatabase.from_uri(f"{get_connection_string()}")
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        extra_tools=[rag_tool],
        prefix=SQL_AGENT_PREFIX,
        agent_executor_kwargs={"handle_parsing_errors": True},
    )

    log.header(
        "Welcome to the interactive Q&A session! Type 'exit' to end the session."
    )

    while True:
        # Prompt the user for a question
        question = input("Please ask a question or type 'exit' to leave: ")

        # Check if the user wants to exit
        if question.lower() == "exit":
            print("Goodbye!")
            break

        log.info(
            f"Asking a question: {question}",
        )

        # Invoke the conversational retrieval chain with the user's question
        agent_executor.invoke({"input": question})


if __name__ == "__main__":
    main()
