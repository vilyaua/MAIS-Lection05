from pydantic import SecretStr
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_key: SecretStr
    model_name: str = "gpt-5.2"

    # Web search
    max_search_results: int = 5
    max_url_content_length: int = 5000

    # RAG
    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "data"
    index_dir: str = "index"
    chunk_size: int = 500
    chunk_overlap: int = 100
    retrieval_top_k: int = 10
    rerank_top_n: int = 3

    # Agent
    output_dir: str = "output"
    max_iterations: int = 10

    model_config = {"env_file": ".env"}


SYSTEM_PROMPT = """"""
