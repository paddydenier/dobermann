from dataclasses import dataclass
import re

_SENTENCE_END = re.compile(r"(?<=[.!?])(?:[\"']+)?\s+(?=[A-Z0-9])")


@dataclass
class Document:
    sentences: list[str]

    @classmethod
    def from_text(cls, text: str) -> "Document":
        sentences = _split_sentences(text)
        return cls(sentences=sentences)


def _split_sentences(text: str) -> list[str]:
    text = text.strip()

    if not text:
        return []

    return [
        sentence.strip() for sentence in _SENTENCE_END.split(text) if sentence.strip()
    ]
