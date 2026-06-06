"""Paragraph-aware word-budget splitter. Coref runs on the whole document first (antecedents
cross chunks), so this only bounds each chunk to the ner context window. Words are a practical
proxy for tokens, like the summarizer's extractive pre-step."""
import os
import re


def _hard_split(words: list[str], max_words: int) -> list[str]:
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def split(text: str, max_words: int | None = None) -> list[str]:
    max_words = max_words or int(os.environ.get("RELATIONS_CHUNK_TOKENS", "512"))
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0

    def flush():
        nonlocal cur, cur_words
        if cur:
            chunks.append("\n\n".join(cur))
            cur, cur_words = [], 0

    for p in paragraphs:
        n = len(p.split())
        if n > max_words:                      # paragraph alone exceeds budget → hard-split
            flush()
            chunks.extend(_hard_split(p.split(), max_words))
            continue
        if cur_words + n > max_words:          # adding it would overflow the current chunk
            flush()
        cur.append(p)
        cur_words += n
    flush()
    return chunks
