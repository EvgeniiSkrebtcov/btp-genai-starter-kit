import sys
import logging

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.langchain import LangchainEmbedding

from .factory import (
    create_llm_and_embeddings,
)

from .helper import print_preview

log = logging.getLogger(__name__)


# Split the docs into chunks
def recursive_split_docs_into_chunks(
    documents: list[Document], chunk_size: int = 256, chunk_overlap: int = 0, **options
):
    try:
        preview = options.get("preview", False)
        log.info("Start splitting documents into chunks using recursive splitter")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        log.success(f"Split {len(documents)} documents into {len(chunks)} chunks.")

        if preview is True:
            print_preview(chunks, "Recursive Splitter")

        return chunks
    except Exception as e:
        log.error(f"Error occurred while splitting documents: {str(e)}")
        sys.exit()


# Split the docs into chunks
def markdown_split_docs_into_chunks(documents: list[Document], **options):
    try:
        preview = options.get("preview", False)

        log.info("Start splitting documents into chunks using markdown splitter")
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )

        md_header_splits = []
        for text_document in documents:
            split_results = markdown_splitter.split_text(text_document.page_content)
            # merge page and chunk metadata
            for split_result in split_results:
                split_result.metadata = {
                    **text_document.metadata,
                    **split_result.metadata,
                }
            md_header_splits.extend(split_results)

        log.success(
            f"Split {len(documents)} documents into {len(md_header_splits)} chunks."
        )

        if preview is True:
            print_preview(md_header_splits, "Markdown Header Splitter")

        return md_header_splits
    except Exception as e:
        log.error(f"Error occurred while splitting documents: {str(e)}")
        sys.exit()


# Split the docs into chunks
def semantic_split_docs_into_chunks(documents: list[Document], **options):
    try:
        preview = options.get("preview", False)

        log.info("Start splitting documents into chunks using semantic splitter")
        _, embeddings = create_llm_and_embeddings()
        embed_model = LangchainEmbedding(embeddings)

        splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
        )

        nodes = splitter.get_nodes_from_documents(documents)

        log.success(f"Split {len(documents)} documents into {len(nodes)} chunks using.")

        if preview:
            print_preview(nodes, "Semantic Splitter")

        return nodes
    except Exception as e:
        log.error(f"Error occurred while splitting documents: {str(e)}")
        sys.exit()
