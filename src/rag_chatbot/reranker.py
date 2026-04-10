from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def cross_encoder_rerank(
    matches: list[dict],
    question: str,
    top_k: int,
    model_name: str,
) -> list[dict]:
    if not matches:
        return []

    try:
        model = _load_model(model_name)
        pairs = [(question, match.get("text", "")) for match in matches]
        scores = model.predict(pairs)
    except Exception:
        return matches[:top_k]

    ranked: list[tuple[float, dict]] = []
    for score, match in zip(scores, matches):
        enriched = dict(match)
        enriched["cross_encoder_score"] = float(score)
        ranked.append((float(score), enriched))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [match for _, match in ranked[:top_k]]
