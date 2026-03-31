import json
from dataclasses import dataclass
from enum import Enum


class DataSet(Enum):
    WIKI_1024 = "datasets/ds_Wiki-1024.json"
    SMAN = "datasets/ds_Sman.json"
    PHILPAPERS_AI = "datasets/ds_PhilPapersAI.json"
    MANIFESTO = "datasets/ds_Manifesto.json"
    CHOI = "datasets/ds_Choi.json"
    ABSTRACTS = "datasets/ds_Abstracts.json"


@dataclass
class Sample:
    sentences: list[str]
    segment_lengths: list[int]


class DataHandler:
    def __init__(self, data_set: DataSet, transform_fn=None):

        with open(data_set.value, "r") as f:
            raw_samples = json.load(f)

        self.samples = [
            Sample(sentences=s["text"], segment_lengths=s["segment_length"])
            for s in raw_samples
        ]

    # map might be more important for larger datasets
    # for now ignore and always batch process
    # !! transform should be handled by the algorithm, not the data handler
