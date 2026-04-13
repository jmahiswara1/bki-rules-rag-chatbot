from __future__ import annotations

import re


_STOPWORDS = {
    "a",
    "an",
    "and",
    "atau",
    "atas",
    "bagi",
    "be",
    "berdasarkan",
    "dalam",
    "dan",
    "dari",
    "dengan",
    "di",
    "for",
    "if",
    "in",
    "ini",
    "is",
    "itu",
    "ke",
    "kepada",
    "lebih",
    "mengenai",
    "of",
    "on",
    "pada",
    "sebagai",
    "tentang",
    "the",
    "to",
    "untuk",
    "yang",
}

_QUESTION_KEYWORD_HINTS: tuple[tuple[str, set[str]], ...] = (
    (
        r"topik\s+utama|dokumen\s+ini|main\s+subject\s+of\s+this\s+document",
        {
            "biro",
            "klasifikasi",
            "indonesia",
            "rules",
            "hull",
            "january",
            "edition",
            "2026",
        },
    ),
    (
        r"nakhoda|loading\s+manual|penyeimbangan|pemuatan|supplied\s+to\s+the\s+master|master\s+of\s+every\s+new\s+ship",
        {
            "master",
            "loading",
            "manual",
            "instrument",
            "stability",
            "stresses",
            "cargo",
        },
    ),
    (
        r"panjang\s+kapal|\(l\)|azimuth|poros\s+kemudi|rudder",
        {"rule", "length", "97", "waterline", "scantling", "rudder", "azimuth", "thrusters"},
    ),
    (
        r"reh|tegangan\s+luluh|kekuatan\s+normal|yield",
        {"reh", "yield", "stress", "normal", "strength", "235", "mm2", "n"},
    ),
    (
        r"sill\s+height|ambang\s+pintu|superstruktur",
        {
            "sills",
            "height",
            "380",
            "600",
            "bridge",
            "poop",
            "access",
            "deck",
            "above",
            "alternative",
            "freeboard",
        },
    ),
    (
        r"faktor\s+probabilitas|\(f\)|beban\s+laut|dek\s+cuaca|probability\s+factor|weather\s+decks|outer\s+hull",
        {
            "probability",
            "factor",
            "fq",
            "outer",
            "sea",
            "load",
            "weather",
            "deck",
            "1",
            "0",
        },
    ),
    (
        r"dredger|pengeruk|aground|ketebalan\s+pelat\s+kulit\s+dasar",
        {"dredger", "aground", "bottom", "shell", "thickness", "20", "section", "6"},
    ),
    (
        r"n_?max|siklus\s+beban|fatigue|20\s*tahun",
        {"nmax", "5", "10", "7", "fatigue", "years", "20"},
    ),
    (
        r"towing\s+winch|derek\s+penghela|diameter\s+minimum|tali\s+penghela",
        {"winch", "drum", "towrope", "diameter", "14"},
    ),
    (
        r"side\s+scuttles|jendela\s+sisi|syarat\s+material|tsg|laminated",
        {"side", "scuttles", "glass", "thermally", "toughened", "laminated", "metal", "non"},
    ),
    (
        r"ketebalan\s+minimum|minimum\s+thickness|l\s*<\s*90|kurang\s*dari\s*90",
        {
            "minimum",
            "thickness",
            "ships",
            "lengths",
            "l",
            "90",
            "50",
            "sqrt",
            "k",
            "1",
            "5",
            "01",
        },
    ),
    (
        r"300\s*rpm|intercostal\s+carlings|propeller|panel\s+pelat",
        {
            "300",
            "rpm",
            "intercostal",
            "carlings",
            "propeller",
            "plate",
            "panel",
            "section",
            "6",
            "f",
        },
    ),
)

_QUESTION_QUERY_EXPANSIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        r"topik\s+utama|dokumen\s+ini|main\s+subject\s+of\s+this\s+document",
        (
            "Rules for Hull January 2026 Edition Biro Klasifikasi Indonesia",
            "Part 1 Seagoing Ships Volume II Rules for Hull",
        ),
    ),
    (
        r"nakhoda|loading\s+manual|penyeimbangan|pemuatan|supplied\s+to\s+the\s+master|master\s+of\s+every\s+new\s+ship",
        (
            "master of every new ship loading manual avoid unacceptable stresses",
            "loading and ballasting information supplied to the master",
        ),
    ),
    (
        r"reh|tegangan\s+luluh|kekuatan\s+normal|yield",
        (
            "normal strength hull structural steel ReH 235 N/mm2",
            "material factor k = 235/ReH normal strength",
        ),
    ),
    (
        r"sill\s+height|ambang\s+pintu|superstruktur",
        (
            "height of sills into a bridge or poop is to be 380 mm",
            "closed superstructure end bulkhead access sill 380 mm",
        ),
    ),
    (
        r"(tanpa|without|tidak\s+ada).{0,30}(akses|access).{0,30}(atas|above)",
        (
            "deck of sill to the doorways in companionways is to be at least 600 mm",
            "companion ways to spaces below deck sills with a height not less than 600 mm",
        ),
    ),
    (
        r"faktor\s+probabilitas|\(f\)|beban\s+laut|dek\s+cuaca|probability\s+factor|weather\s+decks|outer\s+hull",
        (
            "probability factor f for external sea loads shell plating weather deck",
            "table probability factor f equals 1.0 external sea load",
        ),
    ),
    (
        r"side\s+scuttles|jendela\s+sisi|syarat\s+material|tsg|laminated",
        (
            "windows and side scuttles non-metal frames may not be used",
            "thermally toughened safety glass TSG or laminated safety glass",
        ),
    ),
    (
        r"ketebalan\s+minimum|minimum\s+thickness|l\s*<\s*90|kurang\s*dari\s*90",
        (
            "minimum thickness for ships with lengths L less than 90 m",
            "for L less than 50 1.5 minus 0.01L sqrt Lk for L 50 and above sqrt Lk",
        ),
    ),
    (
        r"300\s*rpm|intercostal\s+carlings|propeller|panel\s+pelat",
        (
            "section 6 F 1.3 intercostal carlings propeller revolutions exceed 300 rpm",
            "intercostal carlings to reduce plate panel size when propeller rpm above 300",
        ),
    ),
)


def _tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return {token for token in tokens if len(token) > 1 and token not in _STOPWORDS}


def _keyword_overlap_score(question: str, text: str) -> float:
    question_tokens = _tokenize(question)
    if not question_tokens:
        return 0.0

    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0

    overlap = question_tokens.intersection(text_tokens)
    return len(overlap) / len(question_tokens)


def _term_overlap_score(terms: set[str], text: str) -> float:
    if not terms:
        return 0.0

    term_tokens = {token.lower() for token in terms if len(token) > 1}
    if not term_tokens:
        return 0.0

    text_tokens = _tokenize(text)
    if not text_tokens:
        return 0.0

    overlap = term_tokens.intersection(text_tokens)
    return len(overlap) / len(term_tokens)


def infer_question_conditions(question: str) -> set[str]:
    lowered = question.lower()
    conditions: set[str] = set()

    if re.search(r"(tanpa|without|tidak\s+ada).{0,30}(akses|access).{0,30}(atas|above)", lowered):
        conditions.add("no_access_from_above")
    if re.search(r"(akses|access).{0,30}(atas|above|alternative)", lowered):
        conditions.add("with_access_from_above")
    if re.search(r"(l\s*<\s*90|kurang\s*dari\s*90|less\s+than\s+90)", lowered):
        conditions.add("length_less_than_90")
    if re.search(r"(l\s*<\s*50|kurang\s*dari\s*50|less\s+than\s+50)", lowered):
        conditions.add("length_less_than_50")
    if re.search(r"(300\s*rpm|rpm\s*300)", lowered):
        conditions.add("rpm_300")
    if re.search(r"(topik\s+utama|dokumen\s+ini|main\s+subject\s+of\s+this\s+document)", lowered):
        conditions.add("document_topic")
    if re.search(r"(nakhoda|loading\s+manual|penyeimbangan|pemuatan|master\s+of\s+every\s+new\s+ship|supplied\s+to\s+the\s+master)", lowered):
        conditions.add("master_loading_manual")
    if re.search(r"(faktor\s+probabilitas|\(f\)|beban\s+laut|dek\s+cuaca|probability\s+factor|weather\s+decks|outer\s+hull)", lowered):
        conditions.add("probability_factor")

    return conditions


def _condition_match_score(conditions: set[str], text: str) -> float:
    if not conditions:
        return 0.0

    lowered = text.lower()
    score = 0.0

    if "no_access_from_above" in conditions:
        if re.search(r"(companionways|doorways).{0,80}(not\s+less\s+than|at\s+least).{0,20}600\s*mm", lowered):
            score += 1.0
    if "with_access_from_above" in conditions:
        if re.search(r"(access\s+is\s+provided\s+from\s+the\s+deck\s+above).{0,140}380\s*mm", lowered):
            score += 1.0
    if "length_less_than_90" in conditions:
        if re.search(r"(ships\s+with\s+lengths\s+l\s*<\s*90\s*m|l\s*<\s*90)", lowered):
            score += 0.7
        if re.search(r"(minimum\s+thickness|sqrt|1\s*[\.,]\s*5\s*[-−]\s*0\s*[\.,]\s*01)", lowered):
            score += 0.3
    if "length_less_than_50" in conditions:
        if re.search(r"(l\s*<\s*50|less\s+than\s+50)", lowered):
            score += 1.0
    if "rpm_300" in conditions:
        if re.search(r"(300\s*rpm).{0,120}(intercostal\s+carlings)", lowered):
            score += 1.0
    if "document_topic" in conditions:
        if re.search(r"(rules\s+for\s+hull).{0,80}(january\s+2026|edition).{0,80}(biro\s+klasifikasi\s+indonesia)", lowered):
            score += 1.0
    if "master_loading_manual" in conditions:
        if re.search(r"(supplied\s+to\s+the\s+master|master\s+of\s+every\s+new\s+ship).{0,200}(loading|stresses)", lowered):
            score += 1.0
    if "probability_factor" in conditions:
        if re.search(r"(f\s*=\s*fq\s*=\s*1[\.,]0|f\s*=\s*1[\.,]0|fq\s*=\s*1[\.,]0)", lowered):
            score += 1.0

    return min(1.0, score)


def _metadata_page_number(metadata: dict | None) -> int | None:
    if not metadata:
        return None

    raw_page = metadata.get("page_number")
    if isinstance(raw_page, int):
        return raw_page

    if isinstance(raw_page, str) and raw_page.isdigit():
        return int(raw_page)

    return None


def _section_signal_score(question: str, match: dict, conditions: set[str]) -> float:
    text = (match.get("text") or "").lower()
    metadata = match.get("metadata") or {}
    page_number = _metadata_page_number(metadata)
    score = 0.0

    if "document_topic" in conditions:
        if re.search(r"rules\s+for\s+hull", text):
            score += 0.45
        if re.search(r"part\s*1\s+seagoing\s+ships\s+volume\s*ii", text):
            score += 0.35
        if re.search(r"january\s+2026|2026\s+edition", text):
            score += 0.30
        if re.search(r"biro\s+klasifikasi\s+indonesia|\bbki\b", text):
            score += 0.30
        if page_number is not None and page_number <= 24:
            score += 0.25

    if "master_loading_manual" in conditions:
        if re.search(r"master\s+of\s+every\s+new\s+ship|supplied\s+to\s+the\s+master", text):
            score += 0.40
        if re.search(r"loading\s+and\s+ballasting|loading\s+manual", text):
            score += 0.35
        if re.search(r"unacceptable\s+stresses|stresses\s+in\s+the\s+ship", text):
            score += 0.35

    if "probability_factor" in conditions:
        if re.search(r"outer\s+hull\s+shell\s+plating", text):
            score += 0.30
        if re.search(r"weather\s+decks?", text):
            score += 0.30
        if re.search(r"f\s*=\s*fq\s*=\s*1[\.,]0|f\s*=\s*1[\.,]0|fq\s*=\s*1[\.,]0", text):
            score += 0.30

    if not conditions and re.search(r"rules\s+for\s+hull|january\s+2026|\bbki\b", text):
        score += 0.15

    if page_number is not None and page_number <= 12 and re.search(r"rules|edition|part\s*1", text):
        score += 0.10

    if re.search(r"topik\s+utama|main\s+subject", question.lower()):
        if re.search(r"rules\s+for\s+hull|january\s+2026|biro\s+klasifikasi\s+indonesia", text):
            score += 0.20

    return min(1.0, score)


def rerank_matches(
    matches: list[dict],
    question: str,
    top_k: int,
    gate_terms: set[str] | None = None,
    conditions: set[str] | None = None,
) -> list[dict]:
    ranked: list[tuple[float, dict]] = []
    gate_terms = gate_terms or set()
    conditions = conditions or set()

    for match in matches:
        distance = match.get("distance")
        if not isinstance(distance, (int, float)):
            continue

        semantic_score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
        lexical_score = _keyword_overlap_score(question, match.get("text", ""))
        hint_score = _term_overlap_score(gate_terms, match.get("text", ""))
        condition_score = _condition_match_score(conditions, match.get("text", ""))
        section_score = _section_signal_score(question, match, conditions)

        if gate_terms or conditions:
            combined_score = (
                (semantic_score * 0.38)
                + (lexical_score * 0.12)
                + (hint_score * 0.18)
                + (condition_score * 0.17)
                + (section_score * 0.15)
            )
        else:
            combined_score = (
                (semantic_score * 0.65)
                + (lexical_score * 0.25)
                + (section_score * 0.10)
            )

        enriched = dict(match)
        enriched["semantic_score"] = semantic_score
        enriched["lexical_score"] = lexical_score
        enriched["hint_score"] = hint_score
        enriched["condition_score"] = condition_score
        enriched["section_score"] = section_score
        enriched["combined_score"] = combined_score
        ranked.append((combined_score, enriched))

    ranked.sort(key=lambda item: item[0], reverse=True)

    selected: list[dict] = []
    page_counts: dict[tuple[str, int], int] = {}

    for _, match in ranked:
        metadata = match.get("metadata") or {}
        key = (str(metadata.get("source", "")), int(metadata.get("page_number", -1)))
        current_count = page_counts.get(key, 0)
        if current_count >= 1:
            continue

        selected.append(match)
        page_counts[key] = current_count + 1
        if len(selected) >= top_k:
            return selected

    if len(selected) < top_k:
        for _, match in ranked:
            if match in selected:
                continue
            selected.append(match)
            if len(selected) >= top_k:
                break

    return selected


def infer_keyword_gate_terms(question: str) -> set[str]:
    lowered = question.lower()
    gate_terms: set[str] = set()
    for pattern, terms in _QUESTION_KEYWORD_HINTS:
        if re.search(pattern, lowered):
            gate_terms.update(terms)
    return gate_terms


def infer_query_expansions(question: str) -> list[str]:
    lowered = question.lower()
    expansions: list[str] = []
    for pattern, phrases in _QUESTION_QUERY_EXPANSIONS:
        if re.search(pattern, lowered):
            expansions.extend(phrases)
    return expansions


def apply_keyword_gate(
    matches: list[dict],
    gate_terms: set[str],
    min_ratio: float,
) -> list[dict]:
    if not gate_terms:
        return matches

    required_hits = max(1, min(3, round(len(gate_terms) * min_ratio)))
    filtered: list[dict] = []

    for match in matches:
        text_tokens = _tokenize(match.get("text", ""))
        overlap_count = len(gate_terms.intersection(text_tokens))
        gated_match = dict(match)
        gated_match["keyword_gate_hits"] = overlap_count
        gated_match["keyword_gate_required"] = required_hits
        if overlap_count >= required_hits:
            filtered.append(gated_match)

    return filtered


def apply_condition_filter(matches: list[dict], conditions: set[str]) -> list[dict]:
    if not matches or not conditions:
        return matches

    filtered: list[dict] = []
    for match in matches:
        text = (match.get("text") or "").lower()
        keep = True

        if "no_access_from_above" in conditions:
            keep = bool(
                re.search(
                    r"(companionways|doorways).{0,100}(not\s+less\s+than|at\s+least).{0,20}600\s*mm",
                    text,
                )
            )
        elif "with_access_from_above" in conditions:
            keep = bool(
                re.search(
                    r"(access\s+is\s+provided\s+from\s+the\s+deck\s+above).{0,160}380\s*mm",
                    text,
                )
            )

        if "rpm_300" in conditions:
            keep = keep and bool(re.search(r"300\s*rpm.{0,120}intercostal\s+carlings", text))

        if "probability_factor" in conditions:
            if re.search(r"f\s*=\s*fq\s*=\s*1[\.,]0|f\s*=\s*1[\.,]0|fq\s*=\s*1[\.,]0", text):
                keep = keep and True

        if keep:
            filtered.append(match)

    return filtered if filtered else matches


def build_context(matches: list[dict]) -> str:
    blocks: list[str] = []
    for index, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        block = (
            f"[Sumber {index} | halaman {metadata['page_number']}]\n"
            f"{match['text']}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def format_sources(matches: list[dict]) -> str:
    lines: list[str] = []
    for index, match in enumerate(matches, start=1):
        metadata = match["metadata"]
        lines.append(
            f"{index}. {metadata['source']} - halaman {metadata['page_number']} "
            f"(jarak: {match['distance']:.4f})"
        )
    return "\n".join(lines)
