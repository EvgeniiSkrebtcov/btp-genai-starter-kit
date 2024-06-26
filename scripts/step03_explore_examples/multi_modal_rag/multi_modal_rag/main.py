import os
import io
import base64
import sys
import uuid

from logging import getLogger
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema.messages import HumanMessage, AIMessage
from unstructured.partition.pdf import partition_pdf

from utils.hana import get_connection_to_hana_db, teardown_hana_table
from utils.fs import get_script_dir
from utils.logging import initLogger
from utils.env import assert_env
from utils.http import fetch_file
from langchain_community.vectorstores.hanavector import HanaDB
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.output_parser import StrOutputParser

log = getLogger(__name__)
initLogger()


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


def load_env():
    try:
        # Load environment variables
        load_dotenv()
        assert_env(
            [
                "OPENAI_API_VERSION",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_KEY",
                "AZURE_DEPLOYMENT",
                "EMBEDDING_MODEL",
                "TABLE_NAME",
            ]
        )
        return (
            os.environ.get("OPENAI_API_VERSION"),
            os.environ.get("AZURE_DEPLOYMENT"),
            os.environ.get("EMBEDDING_MODEL"),
            os.environ.get("TABLE_NAME"),
        )
    except Exception as e:
        log.error(e)
        sys.exit()


def extract_pdf_data(pdf: io.BytesIO):
    script_dir = get_script_dir(__file__)
    output_path = (os.path.join(script_dir, "data/output"),)

    text_elements = []
    table_elements = []
    image_elements = []

    try:
        # Get elements
        log.info("Extracting elements from the PDF file.")

        raw_pdf_elements = partition_pdf(
            file=pdf,
            # Using pdf format to find embedded image blocks
            extract_images_in_pdf=True,
            # Use layout model (YOLOX) to get bounding boxes (for tables) and find titles
            # Titles are any sub-section of the document
            infer_table_structure=True,
            # Post processing to aggregate text once we have the title
            chunking_strategy="by_title",
            max_characters=6000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            extract_image_block_output_dir=output_path,
        )

        log.info("Elements text, tables and images from PDF.")

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

        log.info("Number of Text Elements: %s", len(text_elements))
        log.info("Number of Table Elements: %s", len(table_elements))
        log.info("Number of Images: %s", len(image_elements))

        return text_elements, table_elements, image_elements

    except Exception as e:
        log.error(f'Extracting information from pdf failed with: "{str(e)}"')
        sys.exit()


def create_summaries(model, table_elements, text_elements, image_elements):
    table_summaries = []
    text_summaries = []
    image_summaries = []

    for i, te in enumerate(table_elements):
        summary = summarize_table(model, te)
        table_summaries.append(summary)
        log.info("%sth element of tables processed.", {i + 1})

    for i, te in enumerate(text_elements):
        summary = summarize_text(model, te)
        text_summaries.append(summary)
        log.info("%sth element of text processed.", {i + 1})

    for i, ie in enumerate(image_elements):
        summary = summarize_image(model, ie)
        image_summaries.append(summary)
        log.info("%sth element of image processed.", {i + 1})

    log.info("Summmary examples:")
    log.info("First Text Summary: %s", text_summaries[0])
    log.info("First Table Summary: %s", table_summaries[0])
    log.info("First Image Summary: %s", image_summaries[0])

    return text_summaries, table_summaries, image_summaries


def create_vectorstore(embeddings, table_name) -> HanaDB:
    try:
        teardown_hana_table(table_name)
        connection_to_hana = get_connection_to_hana_db()

        return HanaDB(
            embedding=embeddings,
            connection=connection_to_hana,
            table_name=table_name,
        )
    except Exception as e:
        log.error(e)
        sys.exit()


def create_retriever(
    vectorstore,
    text_summaries,
    text_elements,
    table_summaries,
    table_elements,
    image_summaries,
    image_elements,
) -> MultiVectorRetriever:
    try:
        log.info("Ingest data for retrieval.")
        store = InMemoryStore()
        id_key = "doc_id"
        # Initialize the retriever
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
            search_kwargs={"k": 10},
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
        log.info("Adding text summaries to the retriever.")
        add_documents_to_retriever(text_summaries, text_elements)
        log.info("All text summaries added")

        # Add table summaries
        log.info("Adding table summaries to the retriever.")
        add_documents_to_retriever(table_summaries, table_elements)
        log.info("All table summaries added.")

        # Add image summaries
        log.info("Adding image summaries to the retriever.")
        add_documents_to_retriever(image_summaries, image_elements)
        log.info("All image summaries added.")

        log.info("Data ingestion completed.")

        return retriever
    except Exception as e:
        log.error("Ingesting data failed with:%s", e)
        sys.exit()


def main():
    # Initialize the LLM and the embeddings model
    openai_api_version, azure_deployment, embedding_model_name, table_name = load_env()

    chain_gpt_4_vision = AzureChatOpenAI(
        openai_api_version=openai_api_version,
        azure_deployment=azure_deployment,
        temperature=0,
    )

    embeddings = AzureOpenAIEmbeddings(azure_deployment=embedding_model_name)
    vectorstore = create_vectorstore(embeddings, table_name)

    # Fetch the PDF file and extract PDF data
    file_content = fetch_file(
        "https://datasheets.tdx.henkel.com/LOCTITE-HY-4090GY-en_GL.pdf"
    )
    text_elements, table_elements, image_elements = extract_pdf_data(file_content)

    # Create summaries for the elements
    summarized_text, summarized_tables, summarized_images = create_summaries(
        chain_gpt_4_vision,
        text_elements=text_elements,
        table_elements=table_elements,
        image_elements=image_elements,
    )

    # Ingest data for retrieval
    retriever = create_retriever(
        vectorstore=vectorstore,
        text_summaries=summarized_text,
        text_elements=text_elements,
        table_summaries=summarized_tables,
        table_elements=table_elements,
        image_summaries=summarized_images,
        image_elements=image_elements,
    )


#     # -------------------------------------------------------------------------------------
#     # We can retrieve this table
#     template = """Answer the question based only on the following context, which can include text, images and tables:
#     {context}
#     Question: {question}
#     """
#     prompt = ChatPromptTemplate.from_template(template)

#     chain = (
#         {
#             "context": retriever,
#             "question": RunnablePassthrough(),
#         }
#         | prompt
#         | chain_gpt_4_vision
#         | StrOutputParser()
#     )

#     log.header(
#         "Welcome to the interactive Q&A session! Type 'exit' to end the session."
#     )

#     while True:
#         # Prompt the user for a question
#         question = input("Please ask a question or type 'exit' to leave: ")

#         # Check if the user wants to exit
#         if question.lower() == "exit":
#             print("Goodbye!")
#             break

#         log.info(
#             f"Asking a question: {question}",
#         )

#         # Invoke the conversational retrieval chain with the user's question
#         # Output the answer from LLM
#         log.success("Answer from LLM:")
#         result = chain.invoke(question)
#         print(result)


if __name__ == "__main__":
    main()
