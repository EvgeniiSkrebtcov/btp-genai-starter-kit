# Example: Rewrite-Retrieve-Read with SAP BTP Services

In many GenAI cases users are allowed to formulate questions in a free form. That can lead to ambiguity in the question that LLMs can not understand. There are two possible outcomes out of it: LLM will guess the meaning of the query and possibly hallucinate or it will answer something like 'I don't know'.
This example shows how to use the SAP BTP services to implement query rewriting to reduce query ambiguity and improve the quality of the response.  

## 1. Data Ingestion

We begin with data ingestion. This example uses LangChain to load sample documents that will be used for grounding the LLM responses. Document chunks and embedding vectors are then stored in SAP HANA Cloud Vector Engine using the Langchain Vector store adapter.

You can proceed with running the script `ingest_data.py`:
> `python ingest_data.py`

## 2. Rewrite-Retrieve-Read

Then we demonstrate *Rewrite-Retrieve-Read* with SAP HANA Cloud vector engine and SAP GenAI Hub.
We will state ambiguous questions and see how the query rewriting service can help to improve the quality of the response. We will use langchain to retrieve the relevant documents and then use the SAP BTP services to rewrite the query. The rewritten query will be used to retrieve the relevant documents again and then the response will be generated.

You can proceed with running the script `rewrite_retrieve_read.py`:
> `python rewrite_retrieve_read.py`
