import logging
from typing import Optional

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from library.constants.folders import FILE_ENV
from library.data.hana_db import get_connection_to_hana_db
from library.util.logging import initLogger
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from library.constants.table_names import TABLE_NAME
from langchain_community.vectorstores.hanavector import HanaDB
from langchain_core.prompts import ChatPromptTemplate


log = logging.getLogger(__name__)
initLogger()


def main():
    # Load environment variables
    load_dotenv(dotenv_path=str(FILE_ENV), verbose=True)

    db, llm = setup_components()
    question = "What is the summary of the episode 65?"

    log.header("Without database filtering based on user query")
    qaDocumentsWithFilter(db, llm, question)

    log.header("With database filtering based on user query")
    podcast = extract_podcast_title_from_question(llm, question)

    advanced_db_filter = {
        "title": {"$like": f"%{podcast.title()}%"}
    }
    qaDocumentsWithFilter(db, llm, question, advanced_db_filter)

    log.success("""Self querying completed successfully! 
                Now try to ask 'What is the summary of the episode 67?' and notice how self-querying can help to avoid hallucination.
                """)


# Setup database and language model components
def setup_components():
    proxy_client = get_proxy_client("gen-ai-hub")
    embeddings = OpenAIEmbeddings(
        proxy_model_name="text-embedding-ada-002", proxy_client=proxy_client
    )
    db = HanaDB(
        embedding=embeddings,
        connection=get_connection_to_hana_db(),
        table_name=TABLE_NAME,
    )
    llm = ChatOpenAI(proxy_model_name="gpt-35-turbo", proxy_client=proxy_client)
    log.info("Components setup completed")
    return db, llm


# Extract the podcast title from the question
def extract_podcast_title_from_question(llm, question):
    class Podcast(BaseModel):
        title: Optional[str] = Field(
            default=None, description="The title or the episode number of the podcast"
        )

    # Define the prompt to extract data from user query
    system_template = """
    You are an expert extraction algorithm. 
    Only extract relevant information from the question below. 
    If you do not know the value of an attribute asked to extract, 
    return null for the attribute's value.

    Text: {question}
    """

    prompt = PromptTemplate(
        template=system_template, input_variables=["question"]
    )

    runnable = prompt | llm.with_structured_output(schema=Podcast)
    podcast = runnable.invoke({"question": question})
    print("Podcast Title:", podcast.title)
    return podcast.title


# Generate the answer to the question using retrieved documents and applied filters
def qaDocumentsWithFilter(db, llm, question, advanced_db_filter=None):
    qa_prompt_template = """
    You are an expert in SAP podcasts topics. You are provided multiple context items that are related to the prompt you have to answer.
    Use the following pieces of context to answer the question at the end.

    '''
    {context}
    '''

    Question: {question}
    """

    prompt = PromptTemplate(
        template=qa_prompt_template, input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 5, "filter": advanced_db_filter}),
        return_source_documents=True,
        verbose=True,
        chain_type_kwargs={"prompt": prompt},
    )

    result = qa_chain.invoke({"query": question})

    print("Source Documents:")
    for doc in result["source_documents"]:
        print("Title:", doc.metadata["title"], " Page Number:", doc.metadata["page"])

    print("Result:", result["result"])


if __name__ == "__main__":
    main()
