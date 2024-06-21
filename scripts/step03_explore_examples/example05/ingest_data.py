import os
import io
import base64
import logging
import requests

from library.util.logging import initLogger
from library.constants.folders import FILE_ENV
from langchain.schema.messages import HumanMessage, AIMessage
from unstructured.partition.pdf import partition_pdf
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from library.data.hana_db import get_connection_to_hana_db
from langchain_community.vectorstores.hanavector import HanaDB
from library.constants.table_names import (
    VECTOR_EMBEDDINGS_TABLE_NAME,
)
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
import uuid
from langchain.storage import InMemoryStore
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

log = logging.getLogger(__name__)
initLogger()


def load_pdf_from_url(url):
    # Download the PDF file
    response = requests.get(url)
    pdf_file = io.BytesIO(response.content)
    script_dir = get_script_dir()

    # Save the PDF file to the data folder
    pdf_file_path = os.path.join(script_dir, "data/input.pdf")
    with open(pdf_file_path, "wb") as file:
        file.write(response.content)

    return pdf_file


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Function for text summaries
def summarize_text(llm, text_element):
    prompt = f"Summarize the following text:\n\n{text_element}\n\nSummary:"
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# Function for table summaries
def summarize_table(llm, table_element):
    prompt = f"Summarize the following table:\n\n{table_element}\n\nSummary:"
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# Function for image summaries
def summarize_image(llm_with_vision, encoded_image):
    prompt = [
        AIMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe the contents of this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
            ]
        ),
    ]
    response = llm_with_vision.invoke(prompt)
    return response.content


def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


# This function loads the documents into the HANA DB to get them vectorized and validates the documents are loaded correctly
def main():
    # Load environment variables
    load_dotenv(dotenv_path=str(FILE_ENV), verbose=True)

    script_dir = get_script_dir()
    input_path = os.path.join(script_dir, "data/input.pdf")
    output_path = os.path.join(script_dir, "data/output")
    text_elements = []
    table_elements = []
    image_elements = []

    chain_gpt_4_vision = AzureChatOpenAI(
        openai_api_version="2024-02-15-preview", azure_deployment="4o", max_tokens=4096
    )

    embeddings = AzureOpenAIEmbeddings(azure_deployment="text-embedding-3-large")

    # Load the PDF file to ingest
    load_pdf_from_url("https://datasheets.tdx.henkel.com/LOCTITE-HY-4090GY-en_GL.pdf")

    # Get elements
    raw_pdf_elements = partition_pdf(
        filename=input_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=output_path,
    )

    for element in raw_pdf_elements:
        if "CompositeElement" in str(type(element)):
            text_elements.append(element)
        elif "Table" in str(type(element)):
            table_elements.append(element)

    table_elements = [i.text for i in table_elements]
    text_elements = [i.text for i in text_elements]

    for image_file in os.listdir(output_path):
        if image_file.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(output_path, image_file)
            encoded_image = encode_image(image_path)
            image_elements.append(encoded_image)

    print(f"Table Elements: {len(table_elements)}")
    print(f"Text Elements: {len(text_elements)}")
    print(f"Images: {len(image_elements)}")

    # Summarise text, table and images
    table_summaries = []
    for i, te in enumerate(table_elements):
        summary = summarize_table(chain_gpt_4_vision, te)
        table_summaries.append(summary)
        print(f"{i + 1}th element of tables processed.")

    text_summaries = []
    for i, te in enumerate(text_elements):
        summary = summarize_text(chain_gpt_4_vision, te)
        text_summaries.append(summary)
        print(f"{i + 1}th element of texts processed.")

    image_summaries = []
    for i, ie in enumerate(image_elements):
        summary = summarize_image(chain_gpt_4_vision, ie)
        image_summaries.append(summary)
        print(f"{i + 1}th element of images processed.")
        print(summary)

    # Initialize the vector store and storage layer
    connection_to_hana = get_connection_to_hana_db()
    vectorstore = HanaDB(
        embedding=embeddings,
        connection=connection_to_hana,
        table_name=VECTOR_EMBEDDINGS_TABLE_NAME,
    )
    # store = LocalFileStore(FOLDER_DOCS_RAG_SOURCES)
    store = InMemoryStore()
    id_key = "doc_id"
    # Initialize the retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=store, id_key=id_key
    )

    # Function to add documents to the retriever
    def add_documents_to_retriever(summaries, original_contents):
        doc_ids = [str(uuid.uuid4()) for _ in summaries]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]
        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, original_contents)))

    # Add text summaries
    add_documents_to_retriever(text_summaries, text_elements)

    # Add table summaries
    add_documents_to_retriever(table_summaries, table_elements)

    # Add image summaries
    add_documents_to_retriever(
        image_summaries, image_elements
    )  # hopefully real images soon

    # -------------------------------------------------------------------------------------
    # Validate the documents are loaded correctly
    # -------------------------------------------------------------------------------------
    log.info("Validate the documents are loaded correctly")
    cur = connection_to_hana.cursor()
    cur.execute(f"SELECT * FROM {VECTOR_EMBEDDINGS_TABLE_NAME} LIMIT 1")

    rows = cur.fetchall()
    print(rows[0][0])  # The text
    print(rows[0][1])  # The metadata
    print(
        f"{rows[0][2][:100]}..."
    )  # The vector (printing only first 100 characters as it is quite long)
    cur.close()

    log.success("Data ingestion completed.")

    # -------------------------------------------------------------------------------------
    # We can retrieve this table
    template = """Answer the question based only on the following context, which can include text, images and tables:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | chain_gpt_4_vision
        | StrOutputParser()
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
        # Output the answer from LLM
        log.success("Answer from LLM:")
        result = chain.invoke(question)
        print(result)


if __name__ == "__main__":
    main()
