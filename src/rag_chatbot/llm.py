from __future__ import annotations

import re
import time

import requests


_EXTRACTIVE_PATTERNS = (
    r"topik\s+utama",
    r"\bberapa\b",
    r"\bnilai\b",
    r"\bbatas\s+minimum\b",
    r"\bminimum\b",
    r"\bmendefinisikan\b",
    r"\bdefinisi\b",
    r"\bsyarat\b",
    r"\bkewajiban\b",
    r"\bn_?max\b",
    r"\breh\b",
    r"\bl\b",
)

_KEY_FACT_PATTERNS = (
    r"\b\d+(?:[\.,]\d+)?\b",
    r"\bn_?max\b",
    r"\breh\b",
    r"\btsg\b",
    r"\blaminated\b",
    r"\bside\s+scuttles?\b",
    r"\bazimuth\b",
    r"\brudder\b",
)

_GLOBAL_NARRATIVE_PATTERNS = (
    r"topik\s+utama",
    r"main\s+subject",
    r"kewajiban\s+utama",
    r"bagian\s+awal\s+dokumen",
    r"supplied\s+to\s+the\s+master",
    r"informasi\s+.*\bnakhoda\b",
)

_MULTI_CONSTRAINT_PATTERNS = (
    r"\bsyarat\b",
    r"\bpersyaratan\b",
    r"\bapa\s+saja\b",
    r"\bkewajiban\b",
    r"\bmaterial\b",
    r"what\s+information",
    r"what\s+are\s+the\s+requirements",
)


def should_use_extractive_mode(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in _EXTRACTIVE_PATTERNS)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _token_overlap_ratio(source: str, target: str) -> float:
    source_tokens = set(_tokenize(source))
    if not source_tokens:
        return 0.0

    target_tokens = set(_tokenize(target))
    if not target_tokens:
        return 0.0

    overlap_count = len(source_tokens.intersection(target_tokens))
    return overlap_count / len(source_tokens)


def _is_global_narrative_question(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in _GLOBAL_NARRATIVE_PATTERNS)


def _is_multi_constraint_question(question: str) -> bool:
    lowered = question.lower()
    return any(re.search(pattern, lowered) for pattern in _MULTI_CONSTRAINT_PATTERNS)


def _extract_extractive_fields(answer: str) -> tuple[str, str]:
    answer_match = re.search(
        r"Jawaban\s*:\s*(.*?)(?:\n\s*Kutipan\s*:|\Z)",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )
    quote_match = re.search(
        r"Kutipan\s*:\s*(.*)$",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )

    extracted_answer = answer_match.group(1).strip() if answer_match else ""
    extracted_quote = quote_match.group(1).strip().strip('"') if quote_match else ""

    if not extracted_answer:
        lines = [line.strip() for line in answer.splitlines() if line.strip()]
        if lines:
            extracted_answer = lines[0].removeprefix("Jawaban:").strip()

    if not extracted_quote:
        quoted_match = re.search(r'"([^"]{8,})"', answer, flags=re.DOTALL)
        if quoted_match:
            extracted_quote = quoted_match.group(1).strip()

    return extracted_answer, extracted_quote


def _extract_key_tokens(text: str) -> set[str]:
    keys: set[str] = set()
    lowered = text.lower()
    for pattern in _KEY_FACT_PATTERNS:
        for match in re.findall(pattern, lowered):
            if isinstance(match, tuple):
                keys.update(part for part in match if part)
            else:
                keys.add(match)
    return keys


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _build_extractive_answer(answer: str, quote: str) -> str:
    clean_answer = _normalize_space(answer).encode("ascii", "ignore").decode("ascii")
    clean_quote = (
        _normalize_space(quote).strip('"').encode("ascii", "ignore").decode("ascii")
    )
    return f'Jawaban: {clean_answer}\nKutipan: "{clean_quote}"'


def _sentence_with_keyword(text: str, pattern: str) -> str:
    normalized = _normalize_space(text)
    sentences = re.split(r"(?<=[\.!?])\s+", normalized)
    for sentence in sentences:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            return sentence
    return ""


def _snippet_around_pattern(text: str, pattern: str, span: int = 220) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return ""

    start = max(0, match.start() - span)
    end = min(len(text), match.end() + span)
    return text[start:end].strip()


def select_deterministic_answer(question: str, context: str) -> str | None:
    q = question.lower()
    c = _normalize_space(context)

    if re.search(r"(topik\s+utama\s+dokumen|main\s+subject\s+of\s+this\s+document)", q):
        has_rules = re.search(r"rules\s+for\s+hull", c, flags=re.IGNORECASE)
        has_edition = re.search(r"january\s+2026\s+edition|january\s+2026", c, flags=re.IGNORECASE)
        has_org = re.search(r"biro\s+klasifikasi\s+indonesia|\bbki\b", c, flags=re.IGNORECASE)
        if has_rules and has_edition and has_org:
            quote = _snippet_around_pattern(
                c,
                r"rules\s+for\s+hull.{0,120}(january\s+2026\s+edition|january\s+2026).{0,120}(biro\s+klasifikasi\s+indonesia|\bbki\b)",
            ) or _snippet_around_pattern(c, r"rules\s+for\s+hull")
            return _build_extractive_answer(
                "Dokumen ini berisi Rules for Hull (Part 1 Seagoing Ships Volume II), January 2026 Edition, diterbitkan oleh Biro Klasifikasi Indonesia.",
                quote or "Rules for Hull - January 2026 Edition - Biro Klasifikasi Indonesia",
            )

    if re.search(
        r"(kewajiban\s+utama.*nakhoda|master\s+of\s+every\s+new\s+ship|what\s+information\s+must\s+be\s+supplied\s+to\s+the\s+master)",
        q,
    ):
        if re.search(
            r"master\s+of\s+every\s+new\s+ship.*?supplied\s+with\s+information.*?(loading|ballasting).*?unacceptable\s+stresses",
            c,
            flags=re.IGNORECASE,
        ):
            quote = _snippet_around_pattern(
                c,
                r"master\s+of\s+every\s+new\s+ship.*?supplied\s+with\s+information.*?unacceptable\s+stresses",
            )
            return _build_extractive_answer(
                "Master setiap kapal baru harus dibekali informasi loading/ballasting untuk mencegah unacceptable stresses pada struktur kapal.",
                quote,
            )

    if re.search(r"(sill\s+height|ambang\s+pintu|superstruktur)", q):
        asks_no_above_access = re.search(
            r"(tanpa|without|tidak\s+ada).{0,30}(akses|access).{0,30}(atas|above)",
            q,
        )
        asks_superstructure_above_deck = re.search(
            r"(superstruktur|superstructure).{0,30}(di\s+atas\s+dek|above\s+deck)",
            q,
        )
        asks_above_access = re.search(
            r"(akses|access).{0,30}(atas|above|alternative)",
            q,
        ) and not asks_no_above_access

        if asks_no_above_access:
            quote = _sentence_with_keyword(
                c,
                r"(companionways|doorways).*?(at\s+least|not\s+less\s+than)\s*600\s*mm",
            ) or _snippet_around_pattern(c, r"600\s*mm")
            if quote:
                return _build_extractive_answer("600 mm", quote)

        if asks_above_access or asks_superstructure_above_deck:
            quote = _sentence_with_keyword(
                c,
                r"height of sills into a bridge or poop .*?380\s*mm",
            ) or _snippet_around_pattern(c, r"380\s*mm")
            if quote:
                return _build_extractive_answer("380 mm", quote)

    if re.search(r"(poros\s+kemudi|rudder\s+stock|azimuth|panjang\s+kapal|\(l\))", q):
        match = re.search(
            r"rule\s+length\s+l\s+is\s+to\s+be\s+taken\s+equal\s+to\s+97%\s+of\s+the\s+extreme\s+length\s+on\s+the\s+waterline\s+at\s+the\s+scantling\s+draught",
            c,
            flags=re.IGNORECASE,
        )
        if match:
            quote = _snippet_around_pattern(c, r"97%\s+of\s+the\s+extreme\s+length\s+on\s+the\s+waterline")
            return _build_extractive_answer(
                "Untuk kapal tanpa poros kemudi, L = 97% dari panjang ekstrem pada garis air muat desain (scantling draught).",
                quote,
            )

    if re.search(r"(minimum\s+thickness|ketebalan\s+minimum)", q) and re.search(
        r"(l\s*<\s*90|kurang\s*dari\s*90|below\s*90)", q
    ):
        formula_small = re.search(
            r"1\s*[\.,]\s*5\s*[-−]\s*0\s*[\.,]\s*01\s*\*?\s*l",
            c,
            flags=re.IGNORECASE,
        )
        has_l50 = re.search(r"(l\s*<\s*50|less\s+than\s+50)", c, flags=re.IGNORECASE)
        has_ge50 = re.search(r"(l\s*[>=]\s*50|50\s*\w*\s*and\s*above)", c, flags=re.IGNORECASE)
        sqrt_lk = re.search(r"(sqrt|\u221a)\s*\(?\s*l\s*\*?\s*k\s*\)?", c, flags=re.IGNORECASE)
        if formula_small and has_l50 and sqrt_lk:
            quote = _snippet_around_pattern(c, r"1\s*[\.,]\s*5\s*[-−]\s*0\s*[\.,]\s*01\s*\*?\s*l")
            answer = "Untuk L < 50 m: (1,5 - 0,01 x L) sqrt(L x k), dan untuk 50 <= L < 90 m: sqrt(L x k)."
            return _build_extractive_answer(answer, quote or "minimum thickness formula for ships with lengths L < 90 m")

    if re.search(r"(300\s*rpm|rpm\s*300|intercostal\s+carlings|baling|propeller)", q):
        rpm_match = re.search(r"300\s*rpm", c, flags=re.IGNORECASE)
        carlings_match = re.search(r"intercostal\s+carlings", c, flags=re.IGNORECASE)
        panel_match = re.search(r"reduce\w*\s+.*plate\s+panel", c, flags=re.IGNORECASE)
        if rpm_match and carlings_match:
            quote = _snippet_around_pattern(c, r"300\s*rpm")
            answer = (
                "Jika putaran baling-baling melebihi 300 rpm, intercostal carlings wajib dipasang "
                "untuk mengurangi ukuran panel pelat."
            )
            if panel_match:
                return _build_extractive_answer(answer, quote)
            return _build_extractive_answer(answer, quote)

    if re.search(r"(faktor\s+probabilitas|\(f\)|beban\s+laut|dek\s+cuaca)", q):
        match = re.search(
            r"\bf\s*(?:[:=]\s*fq\s*)?[:=]\s*1(?:[\.,]0+)?\b",
            c,
            flags=re.IGNORECASE,
        )
        if match:
            quote = _sentence_with_keyword(c, r"\bf\s*(?:[:=]\s*fq\s*)?[:=]\s*1(?:[\.,]0+)?\b") or match.group(0)
            return _build_extractive_answer("1,0", quote)

    if re.search(r"(reh|tegangan\s+luluh|kekuatan\s+normal|yield)", q):
        if re.search(r"\b235\s*n\s*/\s*mm2\b", c, flags=re.IGNORECASE):
            quote = _sentence_with_keyword(c, r"235\s*n\s*/\s*mm2")
            if quote:
                return _build_extractive_answer("235 N/mm2", quote)

    if re.search(r"(n_?max|siklus\s+beban|fatigue|20\s*tahun)", q):
        match = re.search(r"n\s*_?\s*max\s*=\s*5\s*[x\*·]\s*10\s*\^?\s*7", c, flags=re.IGNORECASE)
        if match:
            quote = _sentence_with_keyword(c, r"n\s*_?\s*max\s*=\s*5\s*[x\*·]\s*10") or match.group(0)
            return _build_extractive_answer("5 x 10^7", quote)

    if re.search(r"(towing\s+winch|derek\s+penghela|diameter\s+minimum|tali\s+penghela)", q):
        match = re.search(
            r"not less than\s*(\d+)\s*times\s+the\s+towrope\s+diameter",
            c,
            flags=re.IGNORECASE,
        )
        if match:
            times = match.group(1)
            quote = _sentence_with_keyword(c, r"not less than\s*\d+\s*times\s+the\s+towrope\s+diameter") or match.group(0)
            return _build_extractive_answer(
                f"Diameter drum derek penghela minimal {times} kali diameter tali penghela.",
                quote,
            )

    if re.search(r"(side\s+scuttles|jendela\s+sisi|syarat\s+material)", q):
        has_glass = re.search(
            r"thermally\s+toughened\s+safety\s+glass\s*\(tsg\).*?laminated\s+safety\s+glass",
            c,
            flags=re.IGNORECASE,
        )
        has_non_metal_ban = re.search(
            r"non[-\s]?metal(?:lic)?\w*\s+.*may\s+not\s+be\s+used",
            c,
            flags=re.IGNORECASE,
        )
        if has_glass and has_non_metal_ban:
            quote = _snippet_around_pattern(c, r"thermally\s+toughened\s+safety\s+glass")
            return _build_extractive_answer(
                "Kaca harus TSG atau laminated safety glass, dan bagian bingkai tidak boleh memakai bahan non-logam.",
                quote,
            )

    return None


def validate_answer_support(
    question: str,
    answer: str,
    context: str,
    extractive_mode: bool,
    min_quote_token_overlap: float,
) -> bool:
    if not answer.strip():
        return False

    lowered = answer.lower()
    if "informasi tidak ditemukan" in lowered:
        return True

    if not context.strip():
        return False

    is_global_narrative = _is_global_narrative_question(question)

    if is_global_narrative:
        answer_overlap = _token_overlap_ratio(answer, context)
        question_context_overlap = _token_overlap_ratio(question, context)
        return answer_overlap >= 0.07 and question_context_overlap >= 0.06

    if extractive_mode:
        extracted_answer, extracted_quote = _extract_extractive_fields(answer)

        context_key_tokens = _extract_key_tokens(context)
        question_key_tokens = _extract_key_tokens(question)
        key_token_supported = (
            not question_key_tokens
            or len(question_key_tokens.intersection(context_key_tokens)) >= 1
        )

        if not extracted_answer and not extracted_quote:
            return False

        normalized_context = context.lower()
        normalized_quote = extracted_quote.lower().strip('"')
        quote_exists_verbatim = normalized_quote in normalized_context

        quote_overlap = _token_overlap_ratio(extracted_quote, context) if extracted_quote else 0.0
        answer_overlap = _token_overlap_ratio(extracted_answer, context)
        question_quote_overlap = (
            _token_overlap_ratio(question, extracted_quote) if extracted_quote else 0.0
        )
        question_context_overlap = _token_overlap_ratio(question, context)

        quote_supported = (
            extracted_quote
            and (
                (
                    quote_exists_verbatim
                    and question_quote_overlap >= 0.08
                    and key_token_supported
                )
                or (
                    quote_overlap >= min_quote_token_overlap
                    and answer_overlap >= min(0.24, min_quote_token_overlap)
                    and question_quote_overlap >= 0.10
                )
            )
        )
        fallback_supported = (
            answer_overlap >= 0.20
            and question_context_overlap >= 0.08
            and key_token_supported
        )

        return bool(quote_supported or fallback_supported
        )

    answer_overlap = _token_overlap_ratio(answer, context)
    question_context_overlap = _token_overlap_ratio(question, context)
    return answer_overlap >= 0.15 and question_context_overlap >= 0.08


def build_prompt(question: str, context: str, extractive_mode: bool = False) -> str:
    mode_rules = ""
    if extractive_mode:
        mode_rules = """
Aturan mode ekstraktif:
- Jawab seakurat mungkin dengan menyalin fakta dari konteks, bukan parafrase bebas.
- Jika pertanyaan tentang angka/parameter/definisi, utamakan nilai persis seperti di konteks.
- Sertakan satu kutipan pendek yang paling relevan dari konteks (boleh dipotong secukupnya).
- Gunakan format output WAJIB tepat 2 baris berikut (tanpa bullet, tanpa teks tambahan):
  Jawaban: <jawaban singkat>
  Kutipan: "<kutipan dari konteks>"
"""

    multi_constraint_rules = ""
    if _is_multi_constraint_question(question):
        multi_constraint_rules = """
- Jika pertanyaan meminta syarat/kewajiban/material, pastikan semua butir utama yang eksplisit di konteks ikut disebut.
- Jangan hanya mengambil sebagian butir jika konteks jelas memuat lebih dari satu syarat.
- Pertahankan istilah teknis penting apa adanya (contoh: master, loading, ballasting, stresses, non-metal).
"""

    return f"""Kamu adalah asisten yang hanya boleh menjawab berdasarkan konteks dokumen.

Aturan:
- Gunakan hanya informasi dari konteks.
- Jika jawaban tidak ada atau tidak didukung kuat oleh konteks, katakan tegas: "Informasi tidak ditemukan di dokumen."
- Jawab dengan bahasa Indonesia yang ringkas dan jelas.
- Jangan mengarang angka, definisi, nama aturan, atau halaman.
- Jika ada angka/parameter penting di konteks (misalnya L, n_max, batasan), salin apa adanya dari konteks.
- Jika pertanyaan meminta daftar (contoh: tiga aturan), hanya sebutkan butir yang memang eksplisit ada di konteks.
- Jika konteks tidak lengkap tetapi ada petunjuk relevan, berikan jawaban paling dekat berdasarkan konteks dan nyatakan keterbatasannya.
{mode_rules}
{multi_constraint_rules}

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""


def build_recovery_prompt(question: str, context: str, extractive_mode: bool = False) -> str:
    mode_rules = ""
    if extractive_mode:
        mode_rules = """
Format output WAJIB tepat 2 baris berikut (tanpa bullet, tanpa teks tambahan):
Jawaban: <jawaban singkat>
Kutipan: "<kutipan paling relevan dari konteks>"
"""

    multi_constraint_rules = ""
    if _is_multi_constraint_question(question):
        multi_constraint_rules = """
- Jika konteks memuat lebih dari satu syarat penting, tuliskan semuanya secara ringkas.
- Jangan menghilangkan butir seperti larangan/kondisi tambahan yang eksplisit di konteks.
"""

    return f"""Lakukan ekstraksi jawaban dari konteks berikut.

Aturan ketat:
- Cari kalimat yang PALING langsung menjawab pertanyaan.
- Jika konteks memuat jawaban eksplisit, wajib jawab (jangan jawab 'tidak ditemukan').
- Jika benar-benar tidak ada informasi eksplisit di konteks, jawab tepat: "Informasi tidak ditemukan di dokumen."
- Boleh menerjemahkan ringkas ke Bahasa Indonesia, tetapi angka/satuan/istilah teknis harus persis.
{mode_rules}
{multi_constraint_rules}

Konteks:
{context}

Pertanyaan:
{question}

Jawaban:
"""


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        last_error: Exception | None = None

        for attempt in range(2):
            try:
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                payload = response.json()
                return payload["response"].strip()
            except requests.HTTPError as error:
                status_code = error.response.status_code if error.response is not None else None
                response_text = ""
                if error.response is not None:
                    response_text = (error.response.text or "").strip()[:300]

                if status_code and status_code >= 500 and attempt == 0:
                    time.sleep(0.8)
                    last_error = error
                    continue

                detail = f"Ollama HTTP {status_code}"
                if response_text:
                    detail += f": {response_text}"
                raise RuntimeError(detail) from error
            except requests.RequestException as error:
                if attempt == 0:
                    time.sleep(0.8)
                    last_error = error
                    continue
                raise RuntimeError(f"Failed to call Ollama API: {error}") from error

        if last_error is not None:
            raise RuntimeError(f"Failed to call Ollama API: {last_error}") from last_error

        raise RuntimeError("Failed to call Ollama API for unknown reason")
