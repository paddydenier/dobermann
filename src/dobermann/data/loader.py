import json
from pathlib import Path

from .types import DataSet, Sample


class DataHandler:
    def __init__(self, data_set: DataSet, transform_fn=None):

        base_path = Path(__file__).parent / "datasets"
        file_path = base_path / data_set.value

        with open(file_path, "r") as f:
            raw_samples = json.load(f)

        self.samples: list[Sample] = [
            Sample(sentences=s["text"], segment_lengths=s["segment_length"])
            for s in raw_samples
        ]
