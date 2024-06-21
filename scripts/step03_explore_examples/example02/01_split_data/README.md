# Example: Splitting Data

This example shows various strategies for splitting data into multiple parts to store and use later during the retrieval process to help answer user questions:
1.**Recursive text splitter**: Recursive chunking based on a list of separators. e.g. dot, new line, etc.
1.**Document specific splitter**: Splitting based on the document structure. e.g. splitting based on the titles, sections, etc.
1.**Document+recursive splitter**: Applying the document specific splitter first and then applying the recursive text splitter.
1.**Semantic splitter**: Splitting based on the semantic meaning of the text. e.g. splitting based on the belonging parts of the text to the same topic.

From the listed above strategies, only the semantic splitter utilizes embeddings model and requires connection to the SAP AI Core service. All the other strategies process data using local libraries.

## Ingest and split Data

We begin with data ingestion. This example uses LangChain to load sample documents that will be used for comparison of the different chunking strategies. Document chunks with the different strategies are then created and result printed in the console.

You can proceed with running the script `split_data.py`:
> `python split_data.py`
