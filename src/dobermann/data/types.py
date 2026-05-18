from dataclasses import dataclass
from enum import Enum


@dataclass
class Sample:
    """Single document sample consisting of sentences and segmentation ground truth.

    Attributes:
        sentences: List of raw sentences of document.
        segment_lengths: List of segment lengths as ground truth.
    """

    sentences: list[str]
    segment_lengths: list[int]


class DataSet(Enum):
    """Built-in benchmark datasets for text segmentation experiments."""

    WIKI_1024 = "ds_Wiki-1024.json"
    """Wikipedia-based segmentation dataset."""
    SMAN = "ds_SMan.json"
    """Scientific/manually annotated segmentation dataset."""
    PHILPAPERS_AI = "ds_PhilPapersAI.json"
    """Philosophy abstracts focused on AI-related topics."""
    MANIFESTO = "ds_Manifesto.json"
    """Political manifesto segmentation dataset."""
    CHOI = "ds_Choi.json"
    """Classic synthetic benchmark introduced by Choi."""
    ABSTRACTS = "ds_Abstracts.json"
    """Research abstract segmentation dataset."""
