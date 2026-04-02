from dataclasses import dataclass
from enum import Enum


@dataclass
class Sample:
    sentences: list[str]
    segment_lengths: list[int]


class DataSet(Enum):
    WIKI_1024 = "ds_Wiki-1024.json"
    SMAN = "ds_SMan.json"
    PHILPAPERS_AI = "ds_PhilPapersAI.json"
    MANIFESTO = "ds_Manifesto.json"
    CHOI = "ds_Choi.json"
    ABSTRACTS = "ds_Abstracts.json"
