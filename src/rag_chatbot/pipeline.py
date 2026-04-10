from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from rag_chatbot.chunking import chunk_pages
from rag_chatbot.config import Settings, ensure_directories
from rag_chatbot.embeddings import EmbeddingModel
from rag_chatbot.llm import (
    OllamaClient,
    build_prompt,
    build_recovery_prompt,
    select_deterministic_answer,
    should_use_extractive_mode,
    validate_answer_support,
)
from rag_chatbot.pdf_loader import extract_pdf_pages
from rag_chatbot.retrieval import (
    apply_condition_filter,
    apply_keyword_gate,
    build_context,
    format_sources,
    infer_question_conditions,
    infer_keyword_gate_terms,
    infer_query_expansions,
    rerank_matches,
)
from rag_chatbot.reranker import cross_encoder_rerank
from rag_chatbot.vector_store import ChromaVectorStore


def _trace_top_matches(matches: list[dict], limit: int = 6) -> list[dict]:
    traces: list[dict] = []
    for match in matches[:limit]:
        metadata = match.get("metadata") or {}
        traces.append(
            {
                "id": match.get("id"),
                "page": metadata.get("page_number"),
                "distance": match.get("distance"),
                "semantic_score": match.get("semantic_score"),
                "lexical_score": match.get("lexical_score"),
                "hint_score": match.get("hint_score"),
                "condition_score": match.get("condition_score"),
                "combined_score": match.get("combined_score"),
            }
        )
    return traces


def _append_trace(settings: Settings, payload: dict) -> None:
    if not settings.debug_trace:
        return

    settings.trace_output_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False)
    with settings.trace_output_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _evaluate_expected_terms(answer: str, must_have_terms: list[str]) -> dict:
    lowered_answer = answer.lower()
    missing_terms: list[str] = []
    for term in must_have_terms:
        if term.lower() not in lowered_answer:
            missing_terms.append(term)
    return {
        "matched": len(missing_terms) == 0,
        "missing_terms": missing_terms,
    }


def ingest_pdf(pdf_path: Path, settings: Settings) -> dict:
    ensure_directories()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = extract_pdf_pages(pdf_path)
    if not pages:
        raise ValueError("No text could be extracted from the PDF.")

    chunks = chunk_pages(
        pages=pages,
        source_name=pdf_path.stem,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    embedder = EmbeddingModel(settings.embedding_model)
    embeddings = embedder.encode_documents([chunk.text for chunk in chunks])

    vector_store = ChromaVectorStore(
        persist_directory=str(settings.chroma_dir),
        collection_name=settings.collection_name,
    )
    vector_store.upsert_chunks(chunks, embeddings)

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "source": pdf_path.name,
    }


def answer_question(question: str, settings: Settings) -> dict:
    return answer_question_with_clients(
        question=question,
        settings=settings,
        embedder=EmbeddingModel(settings.embedding_model),
        vector_store=ChromaVectorStore(
            persist_directory=str(settings.chroma_dir),
            collection_name=settings.collection_name,
        ),
        llm=OllamaClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        ),
    )


def answer_question_with_clients(
    question: str,
    settings: Settings,
    embedder: EmbeddingModel,
    vector_store: ChromaVectorStore,
    llm: OllamaClient,
) -> dict:
    trace_payload: dict = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "question": question,
        "reason": "",
    }

    gate_terms = infer_keyword_gate_terms(question)
    conditions = infer_question_conditions(question)

    query_variants = [question]
    if gate_terms:
        query_variants.append(f"{question} {' '.join(sorted(gate_terms))}")
    query_variants.extend(infer_query_expansions(question))

    trace_payload["conditions"] = sorted(conditions)
    trace_payload["query_variants"] = query_variants

    merged_matches: dict[str, dict] = {}
    candidate_top_k = max(settings.top_k, settings.top_k * settings.retrieval_candidate_multiplier)

    lexical_candidates = vector_store.keyword_search(
        query=" ".join(query_variants),
        top_k=candidate_top_k,
    )
    trace_payload["lexical_candidates_count"] = len(lexical_candidates)
    for match in lexical_candidates:
        match_id = str(match.get("id", ""))
        if match_id:
            merged_matches[match_id] = match

    for query_text in query_variants:
        query_embedding = embedder.encode_query(query_text)
        variant_matches = vector_store.similarity_search(query_embedding, top_k=candidate_top_k)
        for match in variant_matches:
            match_id = str(match.get("id", ""))
            if not match_id:
                continue

            existing = merged_matches.get(match_id)
            if not existing:
                merged_matches[match_id] = match
                continue

            old_distance = existing.get("distance")
            new_distance = match.get("distance")
            if isinstance(old_distance, (int, float)) and isinstance(new_distance, (int, float)):
                if new_distance < old_distance:
                    merged_matches[match_id] = match

    matches = list(merged_matches.values())
    trace_payload["merged_matches_count"] = len(matches)
    if not matches:
        trace_payload["reason"] = "no_matches"
        _append_trace(settings, trace_payload)
        return {
            "answer": (
                "Belum ada dokumen yang di-index atau tidak ada context yang cocok. "
                "Jalankan ingest terlebih dahulu."
            ),
            "matches": [],
            "context": "",
        }

    numeric_matches = [
        match
        for match in matches
        if isinstance(match.get("distance"), (int, float))
    ]
    trace_payload["numeric_matches_count"] = len(numeric_matches)

    strict_matches = [
        match
        for match in numeric_matches
        if match["distance"] <= settings.max_retrieval_distance
    ]

    if strict_matches:
        filtered_matches = strict_matches
        trace_payload["threshold_mode"] = "strict"
    else:
        best_distance = min(
            (match["distance"] for match in numeric_matches),
            default=float("inf"),
        )
        adaptive_threshold = min(
            settings.fallback_max_retrieval_distance,
            best_distance + settings.retrieval_distance_margin,
        )
        filtered_matches = [
            match
            for match in numeric_matches
            if match["distance"] <= adaptive_threshold
        ]
        trace_payload["threshold_mode"] = "adaptive"
        trace_payload["adaptive_threshold"] = adaptive_threshold

    trace_payload["filtered_matches_count"] = len(filtered_matches)

    reranked_matches = rerank_matches(
        filtered_matches,
        question=question,
        top_k=max(settings.final_context_k, settings.final_context_k * 3),
        gate_terms=gate_terms,
        conditions=conditions,
    )

    if settings.use_cross_encoder_reranker:
        reranked_matches = cross_encoder_rerank(
            matches=reranked_matches,
            question=question,
            top_k=settings.final_context_k,
            model_name=settings.reranker_model,
        )

    trace_payload["reranked_top"] = _trace_top_matches(reranked_matches)

    final_matches = [
        match
        for match in reranked_matches
        if match.get("lexical_score", 0.0) >= settings.min_keyword_overlap
    ]

    final_matches = apply_condition_filter(final_matches, conditions)

    gated_matches = apply_keyword_gate(
        final_matches,
        gate_terms=gate_terms,
        min_ratio=settings.keyword_gate_min_ratio,
    )
    if gated_matches:
        final_matches = gated_matches

    final_matches = final_matches[: settings.final_context_k]
    trace_payload["final_matches_count"] = len(final_matches)
    trace_payload["final_top"] = _trace_top_matches(final_matches)

    if len(final_matches) < settings.min_context_matches:
        trace_payload["reason"] = "below_min_context_matches"
        _append_trace(settings, trace_payload)
        return {
            "answer": (
                "Informasi tidak ditemukan di dokumen berdasarkan potongan konteks yang "
                "cukup relevan."
            ),
            "matches": numeric_matches[: settings.top_k],
            "context": "",
        }

    context = build_context(final_matches)
    extractive_mode = should_use_extractive_mode(question)

    deterministic_answer = select_deterministic_answer(question, context)
    if deterministic_answer:
        trace_payload["reason"] = "deterministic_answer"
        _append_trace(settings, trace_payload)
        return {
            "answer": deterministic_answer,
            "matches": final_matches,
            "context": context,
        }

    prompt = build_prompt(
        question,
        context,
        extractive_mode=extractive_mode,
    )
    answer = llm.generate(prompt)

    if settings.conservative_mode:
        is_supported = validate_answer_support(
            question=question,
            answer=answer,
            context=context,
            extractive_mode=extractive_mode,
            min_quote_token_overlap=settings.min_quote_token_overlap,
        )
        if not is_supported:
            answer = "Informasi tidak ditemukan di dokumen."
            trace_payload["validator_rejected"] = True

    if (
        answer.strip().lower() == "informasi tidak ditemukan di dokumen."
        and context.strip()
    ):
        recovery_prompt = build_recovery_prompt(
            question,
            context,
            extractive_mode=extractive_mode,
        )
        recovered_answer = llm.generate(recovery_prompt)
        if recovered_answer and recovered_answer.strip().lower() != "informasi tidak ditemukan di dokumen.":
            if not settings.conservative_mode:
                answer = recovered_answer
            else:
                recovered_supported = validate_answer_support(
                    question=question,
                    answer=recovered_answer,
                    context=context,
                    extractive_mode=extractive_mode,
                    min_quote_token_overlap=settings.min_quote_token_overlap,
                )
                if recovered_supported:
                    answer = recovered_answer

    trace_payload["reason"] = "llm_answer"
    trace_payload["answer"] = answer
    _append_trace(settings, trace_payload)

    return {
        "answer": answer,
        "matches": final_matches,
        "context": context,
    }


def ingest_pdf_command() -> None:
    parser = argparse.ArgumentParser(description="Index a PDF into ChromaDB.")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to a PDF file.")
    args = parser.parse_args()

    settings = Settings()
    result = ingest_pdf(args.pdf, settings)

    print("Ingest selesai.")
    print(f"Sumber : {result['source']}")
    print(f"Halaman: {result['pages']}")
    print(f"Chunk  : {result['chunks']}")


def chat_command() -> None:
    settings = Settings()
    embedder = EmbeddingModel(settings.embedding_model)
    vector_store = ChromaVectorStore(
        persist_directory=str(settings.chroma_dir),
        collection_name=settings.collection_name,
    )
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )

    print("Mode chat aktif. Ketik 'exit' untuk keluar.")

    while True:
        question = input("\nPertanyaan > ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Selesai.")
            break

        result = answer_question_with_clients(
            question=question,
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            llm=llm,
        )
        print("\nJawaban:")
        print(result["answer"])
        print("\nSumber:")
        if result["matches"]:
            print(format_sources(result["matches"]))
        else:
            print("Tidak ada sumber yang ditemukan.")


def evaluate_command() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG answers manually.")
    parser.add_argument(
        "--questions",
        required=True,
        type=Path,
        help="Path to a JSON file with evaluation questions.",
    )
    args = parser.parse_args()

    settings = Settings()
    if settings.debug_trace and settings.trace_output_path.exists():
        settings.trace_output_path.unlink()

    embedder = EmbeddingModel(settings.embedding_model)
    vector_store = ChromaVectorStore(
        persist_directory=str(settings.chroma_dir),
        collection_name=settings.collection_name,
    )
    llm = OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )

    items = json.loads(args.questions.read_text(encoding="utf-8"))

    category_stats: dict[str, dict[str, int]] = {}

    for index, item in enumerate(items, start=1):
        result = answer_question_with_clients(
            question=item["question"],
            settings=settings,
            embedder=embedder,
            vector_store=vector_store,
            llm=llm,
        )
        category = item.get("category", "uncategorized")
        stats = category_stats.setdefault(category, {"total": 0, "not_found": 0})
        stats["total"] += 1
        if "informasi tidak ditemukan" in result["answer"].lower():
            stats["not_found"] += 1

        print("=" * 80)
        print(f"Pertanyaan {index}: {item['question']}")
        print(f"Kategori      : {category}")
        print(f"Expected      : {item['expected_answer']}")
        print(f"Generated     : {result['answer']}")

        must_have_terms = item.get("must_have_terms", [])
        if must_have_terms:
            term_eval = _evaluate_expected_terms(result["answer"], must_have_terms)
            if term_eval["matched"]:
                print("Term Check    : PASS")
            else:
                missing = ", ".join(term_eval["missing_terms"])
                print(f"Term Check    : FAIL (missing: {missing})")

        print("Sumber:")
        if result["matches"]:
            print(format_sources(result["matches"]))
        else:
            print("Tidak ada sumber yang ditemukan.")

    if category_stats:
        print("=" * 80)
        print("Ringkasan per kategori:")
        for category, stats in sorted(category_stats.items()):
            total = stats["total"]
            not_found = stats["not_found"]
            print(
                f"- {category}: total={total}, not_found={not_found}, "
                f"found={total - not_found}"
            )
        if settings.debug_trace:
            print(f"Trace disimpan di: {settings.trace_output_path}")
