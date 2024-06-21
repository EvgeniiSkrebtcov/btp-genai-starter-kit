# Example: Self-query with SAP BTP Services

Retrieving relevant document for RAG can be challenging. In this example, we demonstrate how to use the SAP BTP services to implement a self-query technique that enables the retriever to query itself based on the user data before performing similarity search. This allows us to reduce the number of irrelevant documents retrieved and as a result improve the quality of the response and minimize hallucination.

## 1. Data Ingestion

We begin with data ingestion. This example uses LangChain to load sample documents that will be used for grounding the LLM responses. We download two PDF files that contain transcripts for Episode 64 and Episode 65 of SAP Podcast. After that, we will create document chunks and store embedding vectors in SAP HANA Cloud Vector Engine using the Langchain Vector store adapter. Along with the document chunks, we will store the metadata for each document chunk in the SAP HANA Cloud database. The metadata, such as podcast episode title, will be used to filter the documents during the retrieval process.

You can proceed with running the script `ingest_data.py`:
> `python ingest_data.py`

## 2. Retrieval Augmented Generation

Then we demonstrate *Self-Query* with SAP HANA Cloud Vector Engine and SAP GenAI Hub.
First, we will extract title data from the user query. After that, we will use the title data to filter the documents stored in the SAP HANA Cloud Vector Engine during the retrieval process. The retrieved documents will be used to generate responses using the SAP GenAI Hub.

You can proceed with running the script `self_query_on_data.py`:
> `python self_query_on_data.py`
