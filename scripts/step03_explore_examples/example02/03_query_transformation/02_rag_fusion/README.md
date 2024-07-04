# Example: RAG Fusion with SAP BTP Services

In many GenAI cases users are allowed to formulate questions in a free form. That can lead to ambiguity in the question that LLMs can not understand. There are two possible outcomes out of it: LLM will guess the meaning of the query and possibly hallucinate or it will answer something like 'I don't know'.
This example shows how to use the SAP BTP services to implement RAG Fusion technique. With RAG Fusion, we generate multiple queries similar to the user query, retrieve relevant documents for all of them and then combine results together using Reciprocal Rank Fusion algorithm. The final list of documents is then used to generate the response.

## 1. Data Ingestion

We begin with data ingestion. This example uses LangChain to load sample documents that will be used for grounding the LLM responses. Document chunks and embedding vectors are then stored in SAP HANA Cloud Vector Engine using the Langchain Vector store adapter.

You can proceed with running the script `ingest_data.py`:
> `python ingest_data.py`

## 2. RAG Fusion

Then we demonstrate *RAG Fusion* with SAP HANA Cloud vector engine and SAP GenAI Hub.
We use LangChain to generate multiple queries similar to the user query, retrieve relevant documents for all of them and then combine results together using Reciprocal Rank Fusion algorithm. The final list of documents is then used to generate the response.
We will also compare results with and without RAG Fusion.

You can proceed with running the script `rag_fusion.py`:
> `python rag_fusion.py`
