from gen_ai_hub.proxy.langchain.init_models import init_llm
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from .config import EMBEDDINGS_MODEL_NAME, LLM_MODEL_NAME
from langchain_core.rate_limiters import InMemoryRateLimiter


def create_llm_and_embeddings():
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.5,  # We can only make a request once every 5 seconds
        check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
        max_bucket_size=10,  # Controls the maximum burst size.
    )

    llm = init_llm(LLM_MODEL_NAME, max_tokens=300, rate_limiter=rate_limiter)

    embeddings = init_embedding_model(EMBEDDINGS_MODEL_NAME)
    return llm, embeddings
