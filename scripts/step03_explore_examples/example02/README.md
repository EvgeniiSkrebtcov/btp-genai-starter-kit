# Example: Advanced RAG with SAP BTP Services
This series of examples demonstrates how to use the SAP Business Technology Platform (BTP) services to implement various advanced RAG techniques.

## Advanced RAG

Retrieval Augmented Generation (RAG) is a dominant paradigm for generating high-quality responses to user queries by leveraging the knowledge stored in a large corpus of documents. However, this pattern has a number of limitations that are not addressed by the standard RAG implementation.

* Unclear Questions: Occasionally, users ask questions that aren't clearly defined, which can lead to irrelevant information being retrieved.

* Inaccuracy in Retrieval: The documents retrieved may not necessarily relate to the question being asked.

* Incomplete Knowledge Base: The information the user wants to find may not be included in the available knowledge base.

* Performance Restrictions of Context Window: An attempt to retrieve too much information can overload the capacity of the context window, or it might create a context window that's too large to provide a result within a suitable amount of time.

* Inefficient Model Selection: Choosing a non-optimal GenAI model or configuration for a specific problem can lead to compromised quality or increased costs.

These and many more limitations can affect quality and costs of the solution.

Many advanced techniques exist to address these limitations. This example demonstrates how to use the SAP BTP services to implement some of these techniques.

## Overview of the examples

1. **Splitting data**: In this technique, the data split into multiple parts to store and use later during retrieval process to help answer user questions. In the example provided, we will showcase four different chunking strategies and explore the differences: recursive text splitter, document specific splitter, document+recursive splitter and semantic splitter.  
The code for this example is available in the [01_split_data](/scripts/step03_explore_examples/example02/01_split_data/) folder.

1. **Self-query**: This technique enables retriever to query itself based on the user data before performing similarity search. This allows to reduce the number of irrelevant documents retrieved and as a result improve the quality of the response and minimize hallucination.  
The code for this example is available in the [02_self_query](/scripts/step03_explore_examples/example02/02_self_query/ingest_data.py) folder.

1. **Query transformation**: When users asking questions that are not clearly defined, the query transformation technique can help to improve the quality of the response. In this example, we will demonstrate how to use the SAP BTP services to implement two different ways of reducing query ambiguity and improving the quality of the response: query rewriting and RAG Fusion.  
The code for this example is available in the [03_query_transformation](/scripts/step03_explore_examples/example02/03_query_transformation/) folder.
