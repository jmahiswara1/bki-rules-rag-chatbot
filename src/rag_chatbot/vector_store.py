from __future__ import annotations

from dataclasses import asdict
import re

import chromadb
from chromadb.errors import InvalidArgumentError

from rag_chatbot.chunking import TextChunk


class ChromaVectorStore:
    def __init__(self, persist_directory: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def upsert_chunks(self, chunks: list[TextChunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have a matching embedding.")

        try:
            self._collection.upsert(
                ids=[chunk.chunk_id for chunk in chunks],
                documents=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=[asdict(chunk) for chunk in chunks],
            )
        except InvalidArgumentError as error:
            if "dimension" not in str(error).lower():
                raise

            raise ValueError(
                "Dimensi embedding tidak cocok dengan collection Chroma yang sudah ada. "
                "Ini biasanya terjadi setelah mengganti EMBEDDING_MODEL. "
                "Solusi: ganti CHROMA_COLLECTION ke nama baru (mis. pdf_chunks_e5) "
                "atau hapus folder data/chroma lalu ingest ulang."
            ) from error

    def similarity_search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        matches: list[dict] = []
        for doc_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            matches.append(
                {
                    "id": doc_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return matches

    def keyword_search(self, query: str, top_k: int) -> list[dict]:
        if self._collection.count() == 0:
            return []

        query_tokens = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
        query_tokens = {token for token in query_tokens if len(token) > 1}
        if not query_tokens:
            return []

        raw = self._collection.get(include=["documents", "metadatas"])
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        ids = raw.get("ids") or []

        scored: list[tuple[float, dict]] = []
        for doc_id, document, metadata in zip(ids, documents, metadatas):
            text = str(document or "")
            text_tokens = set(re.findall(r"[A-Za-z0-9_]+", text.lower()))
            if not text_tokens:
                continue

            overlap = query_tokens.intersection(text_tokens)
            if not overlap:
                continue

            token_score = len(overlap) / len(query_tokens)
            phrase_bonus = 0.0
            lowered_text = text.lower()
            for token in ("300", "rpm", "intercostal", "carlings", "1.5", "0.01", "sqrt"):
                if token in query.lower() and token in lowered_text:
                    phrase_bonus += 0.05

            score = min(1.0, token_score + phrase_bonus)
            scored.append(
                (
                    score,
                    {
                        "id": doc_id,
                        "text": text,
                        "metadata": metadata,
                        "distance": 1.0 - score,
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        return [match for _, match in scored[:top_k]]
