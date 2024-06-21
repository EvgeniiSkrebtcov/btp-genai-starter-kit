import logging

from dotenv import load_dotenv
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI, OpenAIEmbeddings
from langchain.load import dumps, loads
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

    db, llm = initialize_components()

    template = """
        Answer the question in detail and as truthfully as possible based only on the provided context. If you're unsure of the question or answer, say "Sorry, I don't know".
        <context>
        {context}
        </context>

        Question: {original_query}
    """

    prompt = ChatPromptTemplate.from_template(template)

    retriever = db.as_retriever(k=5)

    original_query = "Cloud Foundry, Kyma or what?"

    log.header("Invoke query without RAG Fusion")
    qaWithoutFusion(llm, original_query, prompt, retriever)

    log.header("Invoke query with RAG Fusion")
    qaWithRagFusion(llm, original_query, prompt, retriever)

    log.success("RAG Fusion successfully completed.")


# Initialize database and language model components
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


# Question answering without RAG Fusion
def qaWithoutFusion(llm, original_query, prompt, retriever):
    chain = (
        {"context": retriever, "original_query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("Query before rewrite: ", original_query)

    result = chain.invoke(original_query)
    print("QA result without query rewrite: ", result)
    return result


# Question answering with RAG Fusion
def qaWithRagFusion(llm, original_query, original_prompt, retriever):
    fusion_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that generates multiple search queries based on a single input query.",
            ),
            ("user", "Generate multiple search queries related to: {original_query}"),
            ("user", "OUTPUT (4 queries):"),
        ]
    )

    def printAndSplit(generatedQueries: str):
        print("Generated queries: ", generatedQueries)
        return generatedQueries.split("\n")

    # Define the pipeline that generates multiple search queries based on a single input query
    chain = (
        {
            "context": {
                "question_to_rewrite": RunnablePassthrough()
                | fusion_prompt
                | llm
                | StrOutputParser()
                | printAndSplit
                | retriever.map()
                | reciprocal_rank_fusion
            },
            "original_query": RunnablePassthrough(),
        }
        | original_prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(original_query)

    print("Result with RAG Fusion: ", result)


# Define the reciprocal rank fusion function that combines the results from multiple retrievers into a single ranked list
def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            fused_scores.setdefault(doc_str, 0)
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results


if __name__ == "__main__":
    main()
