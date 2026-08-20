# create document from dataset
# create document from raw text, document from text file.
# create document from HTML
# create document from URL

from dataclasses import dataclass

import nltk


@dataclass
class Document:
    sentences: list[str]

    @classmethod
    def from_text(cls, text: str) -> "Document":
        sentences = nltk.sent_tokenize(text)

        return cls(sentences=sentences)
