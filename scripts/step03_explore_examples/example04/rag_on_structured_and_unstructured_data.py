from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from library.constants.folders import FILE_ENV
from library.constants.table_names import VECTOR_EMBEDDINGS_TABLE_NAME
from library.constants.prompts import SQL_AGENT_PREFIX
from library.util.logging import initLogger
from langchain_community.vectorstores.hanavector import HanaDB
from langchain.tools import BaseTool
from langchain.sql_database import SQLDatabase
from typing import Optional
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.agents import create_sql_agent
from langchain.agents.types import AgentType
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from library.data.hana_db import get_connection_to_hana_db, get_connection_string
from dotenv import load_dotenv
import logging
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


log = logging.getLogger(__name__)
initLogger()

def main():
    # Load environment variables
    load_dotenv(dotenv_path=str(FILE_ENV), verbose=True)

    # Get the proxy client for the AI Core service
    proxy_client = get_proxy_client("gen-ai-hub")

    # Step 1. Define LLM model
    llm = ChatOpenAI(proxy_model_name="gpt-4", proxy_client=proxy_client, temperature=0)
    log.info("ChatOpenAI object created")

    

    class RAGTool(BaseTool):
        name = "rag_tool"
        description = """
        Useful information about Cities from Wikipedia.
        Input: A question about a city.
        Output: The answer to the question."""

        def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
        ) -> str:
            """Use the tool."""
            connection_to_hana = get_connection_to_hana_db()
            proxy_client = get_proxy_client("gen-ai-hub")
            embeddings = OpenAIEmbeddings(
                proxy_model_name="text-embedding-ada-002", proxy_client=proxy_client
            )
            db = HanaDB(embedding=embeddings, connection=connection_to_hana, table_name=VECTOR_EMBEDDINGS_TABLE_NAME)
            retriever = db.as_retriever(search_kwargs={"k":10})

            system_prompt = (
                "Use the given context to answer the question. "
                "If you don't know the answer, say you don't know. "
                "Use three sentence maximum and keep the answer concise. "
                "Context: {context}"
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )


            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            chain = create_retrieval_chain(retriever, question_answer_chain)

            result = chain.invoke({"input": query})

            return result["answer"]

        async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
        ) -> str:
            """Use the tool asynchronously."""
            raise NotImplementedError("custom_search does not support async")

    rag_tool = RAGTool()
    db = SQLDatabase.from_uri(f"{get_connection_string()}")
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        extra_tools=[rag_tool],
        prefix=SQL_AGENT_PREFIX,
        agent_executor_kwargs={'handle_parsing_errors': True}
    )

    log.header("Welcome to the interactive Q&A session! Type 'exit' to end the session.")  
      
    while True:  
        # Prompt the user for a question  
        question = input("Please ask a question or type 'exit' to leave: ")  
          
        # Check if the user wants to exit  
        if question.lower() == 'exit':  
            print("Goodbye!")  
            break  
  
        log.info(f"Asking a question: {question}", )  
          
        # Invoke the conversational retrieval chain with the user's question  
        agent_executor.invoke(
            {
                "input": question
            }
        )

if __name__ == "__main__":
    main()
