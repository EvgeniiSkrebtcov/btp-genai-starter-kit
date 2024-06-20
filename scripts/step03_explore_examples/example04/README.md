# Example: RAG on SAP BTP with structured and unstructured data
This example shows how to create a RAG application that uses both **structured** and **unstructured** data for grounding LLM responses.

## 1. Data Ingestion
We begin with data ingestion.
This time we use the following data dfor grounding:
 - Create table in SAP HANA DB with Information about City, Population and Country it belongs to.
 - Create table with embeddings of the Wikipedia pages that are related to the Cities. This example uses LangChain adapter for HanaDB Vector Engine to load sample documents that will be used for grounding the LLM responses. 

You can proceed with running the script `ingest_data.py`:
> `python ingest_data.py`

## 2. Retrieval Augmanted Generation
Then we demonstrate *Retrieval Augmented Generation* app that can use both:
- Structured data stored in the tables in SAP HANA Cloud.
- Unstructured data from Wiki via SAP HANA Cloud Vector engine and SAP GenAI Hub.

We implement the application as a LangChain `SQL Agent` that uses SQLAlchemy dialect for SAP HANA for constructing the SQL Queries, and equip it with an additional custom RAG Tool for answering detailed questions about Cities.
The agent Iterates on the question presented by the user and uses REACT pattern choosing which tool to select on every iteration. This ecample requires `gpt-4` model for reliable results.

You can proceed with running the script `rag_on_structured_and_unstructured_data.py`:
> `python rag_on_structured_and_unstructured_data.py`

**Example questions:**
- *Can you give me the country corresponding to each city?* - This question corresponds to the structured data and can be answered by constructing and executing an SQL statement to the `city_stats` table in Hana DB.
- *Tell me about the history of Berlin.* - Answer on this question can not be found in the `city_stats` table and requires a call to the `RAGTool` for accessing relevant Wiki Pages info.
- *Tell me about the arts and culture of the city with the highest population.* - Answer to this question requires both, finding out the city with the highest population and using the `RAGTool` for accessing relevant Wiki Pages info.