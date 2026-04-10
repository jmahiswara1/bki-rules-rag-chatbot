from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz


@dataclass(slots=True)
class PageDocument:
    page_number: int
    text: str


def clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_pages(pdf_path: Path) -> list[PageDocument]:
    pages: list[PageDocument] = []
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document, start=1):
            raw_text = page.get_text("text")
            cleaned = clean_text(raw_text)
            if cleaned:
                pages.append(PageDocument(page_number=page_index, text=cleaned))
    return pages
