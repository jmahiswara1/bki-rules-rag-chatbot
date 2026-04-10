from __future__ import annotations

import logging
import os

from sentence_transformers import SentenceTransformer


def _configure_model_runtime_logs() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        _configure_model_runtime_logs()
        self._model_name = model_name
        self._use_e5_prefix = "e5" in model_name.lower()
        self._model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        documents = texts
        if self._use_e5_prefix:
            documents = [f"passage: {text}" for text in texts]

        embeddings = self._model.encode(documents, normalize_embeddings=True)
        return embeddings.tolist()

    def encode_query(self, text: str) -> list[float]:
        query = text
        if self._use_e5_prefix:
            query = f"query: {text}"

        embedding = self._model.encode(query, normalize_embeddings=True)
        return embedding.tolist()
