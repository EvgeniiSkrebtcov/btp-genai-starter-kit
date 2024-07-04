def print_preview(chunks, header: str, number_of_chunks: int = 2):
    print("=" * 10, header, "=" * 10)
    for i, chunk in enumerate(chunks[:number_of_chunks]):
        print(
            f"chunk {i}: {chunk.text if hasattr(chunk, 'text') else chunk.page_content}"
        )
