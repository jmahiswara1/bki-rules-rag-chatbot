from __future__ import annotations

from dataclasses import dataclass
import re

from rag_chatbot.pdf_loader import PageDocument


@dataclass(slots=True)
class TextChunk:
    chunk_id: str
    text: str
    source: str
    page_number: int
    chunk_index: int


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 140:
        return False

    if re.match(r"^(section|sec\.|chapter|part|pt\.)\b", stripped, flags=re.IGNORECASE):
        return True
    if re.match(r"^\d+(?:\.\d+){0,3}\b", stripped):
        return True

    letters = [char for char in stripped if char.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
    return upper_ratio >= 0.8


def _split_long_text(text: str, max_len: int) -> list[str]:
    if len(text) <= max_len:
        return [text]

    parts: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + max_len)
        if end < len(text):
            split_point = text.rfind(" ", cursor, end)
            if split_point > cursor + (max_len // 2):
                end = split_point

        piece = text[cursor:end].strip()
        if piece:
            parts.append(piece)
        cursor = end

    return parts


def _extract_semantic_units(text: str) -> list[str]:
    paragraphs = [segment.strip() for segment in re.split(r"\n\s*\n+", text) if segment.strip()]
    units: list[str] = []
    pending_heading = ""

    for paragraph in paragraphs:
        first_line = paragraph.splitlines()[0].strip()
        if _is_heading(first_line) and len(paragraph.splitlines()) <= 2:
            pending_heading = paragraph
            continue

        if pending_heading:
            units.append(f"{pending_heading}\n\n{paragraph}")
            pending_heading = ""
        else:
            units.append(paragraph)

    if pending_heading:
        units.append(pending_heading)

    return units


def _tail_overlap(text: str, overlap_size: int) -> str:
    if overlap_size <= 0 or len(text) <= overlap_size:
        return text.strip()

    tail = text[-overlap_size:]
    first_space = tail.find(" ")
    if first_space != -1 and first_space < len(tail) - 1:
        tail = tail[first_space + 1 :]
    return tail.strip()


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be 0 or greater.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    semantic_units = _extract_semantic_units(text)
    units: list[str] = []
    for unit in semantic_units:
        units.extend(_split_long_text(unit, chunk_size))

    chunks: list[str] = []
    current = ""

    for unit in units:
        if not current:
            current = unit
            continue

        candidate = f"{current}\n\n{unit}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current.strip())
        overlap_prefix = _tail_overlap(current, chunk_overlap)
        current = f"{overlap_prefix}\n\n{unit}".strip() if overlap_prefix else unit

        if len(current) > chunk_size:
            forced = _split_long_text(current, chunk_size)
            chunks.extend(forced[:-1])
            current = forced[-1]

    if current.strip():
        chunks.append(current.strip())

    return chunks


def chunk_pages(
    pages: list[PageDocument],
    source_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    chunk_counter = 0

    for page in pages:
        for local_index, chunk_text in enumerate(
            split_text(page.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            chunk_counter += 1
            chunks.append(
                TextChunk(
                    chunk_id=f"{source_name}-p{page.page_number}-c{local_index}",
                    text=chunk_text,
                    source=source_name,
                    page_number=page.page_number,
                    chunk_index=chunk_counter,
                )
            )

    return chunks
