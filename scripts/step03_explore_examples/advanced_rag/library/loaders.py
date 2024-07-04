import sys
from logging import getLogger

from llama_index.core import SimpleDirectoryReader
from langchain_community.document_loaders import GitLoader, PyMuPDFLoader

from .helper import print_preview

log = getLogger(__name__)


def fetch_document_from_git(url="https://github.com/SAP/terraform-provider-btp"):
    try:
        log.info("Loading documents from the GitHub repository")

        loader = GitLoader(
            clone_url=url,
            repo_path="./gen/docs/",
            file_filter=lambda file_path: file_path.endswith(".md"),
            branch="main",
        )

        documents = loader.load()
        log.success("Documents loaded successfully")
        return documents
    except Exception as e:
        log.error(f"Error loading documents from GitHub repository: {e}")
        sys.exit()


def load_from_dir(input_dir="./gen/docs/"):
    try:
        log.info("Loading documents from the directory")

        required_exts = [".md"]

        reader = SimpleDirectoryReader(
            input_dir=input_dir,
            required_exts=required_exts,
            recursive=True,
        )

        docs = reader.load_data()
        log.success(f"Loaded {len(docs)} documents from the directory")
        return docs
    except Exception as e:
        log.error(f"Error loading documents from the directory: {e}")
        sys.exit()


def load_and_split_pdf(url, **options):
    try:
        preview = options.get("preview", False)
        log.info(f"Loading PDF from url: {url}")
        loader = PyMuPDFLoader(url)
        chunks = loader.load_and_split()
        log.success("PDF loaded and split successfully")

        if preview is True:
            print_preview(chunks, "PDF Preview", 1)

        return chunks
    except Exception as e:
        log.error(f"Error loading and splitting PDF: {e}")
        sys.exit()
