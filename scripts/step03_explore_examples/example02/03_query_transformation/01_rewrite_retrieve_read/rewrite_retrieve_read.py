import logging

from dotenv import load_dotenv
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from library.constants.folders import FILE_ENV
from library.constants.table_names import TABLE_NAME
from library.data.hana_db import get_connection_to_hana_db
from library.util.logging import initLogger



log = logging.getLogger(__name__)
initLogger()


def main():
    # Load environment variables
    load_dotenv(dotenv_path=str(FILE_ENV), verbose=True)
    log.header("Rewrite, retrieve, and read")

    db, llm = initialize_components()

    template = """
        Answer the question in detail and as truthfully as possible based only on the provided context. If you're unsure of the question or answer, say "Sorry, I don't know".
        <context>
        {context}
        </context>

        Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    def retriever(query):
        retrieved_chunks = db.similarity_search(query, k=5)
        return retrieved_chunks

    simple_query = "Cloud Foundry, Kyma or what?"

    log.header("QA without query rewrite")
    invokeQueryWithoutRewrite(llm, simple_query, prompt, retriever)

    log.header("QA with query rewrite")
    invokeQueryWithRewrite(llm, simple_query, prompt, retriever)

    log.success("Rewrite, retrieve, and read successfully completed.")


def initialize_components():
    # Get the connection to the HANA DB
    connection_to_hana = get_connection_to_hana_db()
    log.info("Connection to HANA DB established")

    # Get the proxy client for the AI Core service
    proxy_client = get_proxy_client("gen-ai-hub")

    # Create the OpenAIEmbeddings object
    embeddings = OpenAIEmbeddings(
        proxy_model_name="text-embedding-ada-002", proxy_client=proxy_client
    )
    log.info("OpenAIEmbeddings object created")

    llm = ChatOpenAI(proxy_model_name="gpt-35-turbo", proxy_client=proxy_client)
    log.info("ChatOpenAI object created")

    # Create the HanaDB object
    db = HanaDB(
        embedding=embeddings, connection=connection_to_hana, table_name=TABLE_NAME
    )
    return db, llm


def invokeQueryWithoutRewrite(llm, simple_query, prompt, retriever):
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Query before rewrite: ", simple_query)

    result = chain.invoke(simple_query)
    print("QA result without query rewrite: ", result)
    return result


def invokeQueryWithRewrite(llm, simple_query, prompt, retriever):
    template = """Provide a better query for
    the database similarity retrieval and large language model to answer the given question. 
    Question: {question_to_rewrite} 
    Answer:"""
    rewrite_prompt = ChatPromptTemplate.from_template(template)

    def rewritten_query(rewritten_query: str):
        print("Query after rewrite: ", rewritten_query)
        return rewritten_query

    rewriter = rewrite_prompt | llm | StrOutputParser()

    rewrite_retrieve_read_chain = (
        rewriter
        | {
            "context": {"question_to_rewrite": RunnablePassthrough() | retriever},
            "question": RunnablePassthrough() | rewritten_query,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    rewritten_result = rewrite_retrieve_read_chain.invoke(simple_query)
    print("QA result after query rewrite: ", rewritten_result)


if __name__ == "__main__":
    main()
