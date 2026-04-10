from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"


@dataclass(slots=True)
class Settings:
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
    )
    collection_name: str = os.getenv("CHROMA_COLLECTION", "pdf_chunks")
    chroma_dir: Path = CHROMA_DIR
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    top_k: int = int(os.getenv("TOP_K", "8"))
    max_retrieval_distance: float = float(os.getenv("MAX_RETRIEVAL_DISTANCE", "1.10"))
    fallback_max_retrieval_distance: float = float(
        os.getenv("FALLBACK_MAX_RETRIEVAL_DISTANCE", "1.30")
    )
    retrieval_distance_margin: float = float(
        os.getenv("RETRIEVAL_DISTANCE_MARGIN", "0.20")
    )
    min_context_matches: int = int(os.getenv("MIN_CONTEXT_MATCHES", "1"))
    final_context_k: int = int(os.getenv("FINAL_CONTEXT_K", "4"))
    min_keyword_overlap: float = float(os.getenv("MIN_KEYWORD_OVERLAP", "0.00"))
    keyword_gate_min_ratio: float = float(os.getenv("KEYWORD_GATE_MIN_RATIO", "0.20"))
    conservative_mode: bool = os.getenv("CONSERVATIVE_MODE", "1") == "1"
    min_quote_token_overlap: float = float(os.getenv("MIN_QUOTE_TOKEN_OVERLAP", "0.45"))
    retrieval_candidate_multiplier: int = int(
        os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "4")
    )
    use_cross_encoder_reranker: bool = (
        os.getenv("USE_CROSS_ENCODER_RERANKER", "0") == "1"
    )
    reranker_model: str = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    debug_trace: bool = os.getenv("DEBUG_TRACE", "1") == "1"
    trace_output_path: Path = Path(
        os.getenv("TRACE_OUTPUT_PATH", str(PROCESSED_DATA_DIR / "eval_trace.jsonl"))
    )


def ensure_directories() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
