# Example: Multi Modal RAG

The following examples shows how to extract text, tables and images (diagrams) from a given PDF file.

## Prerequisites

> Attention!
> Some libraries e.g. tesseract must be installed on the host environment.
> In case you don't want to install the package on you local machine, use the dev container to get an isolated environment to try out the examples!

To extract information from the PDF multiple packages are required:

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) - for information extraction

## 1. Overview

In the first step tesseract is used to extract information from the PDF. For the tables and images a summary is created. Afterwards embeddings are created for text, images, and tables.

## 2. How to run
You can proceed with running the script `semistructured_multimodal_rag.py`:
> `python semistructured_multimodal_rag.py`

**Example questions:**
- Summarise curation time
- What are the storage conditions?
- summarise Chemical/Solvent Resistance
- How to call Henkel Europe?