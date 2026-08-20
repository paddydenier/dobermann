import json
from pathlib import Path

from .types import DataSet, Sample


class DataHandler:
    @staticmethod
    def samples(data_set: DataSet) -> list[Sample]:
        """Load and parse built-in datasets.

        Args:
            data_set: Dataset identifier used to locate JSON file.

        Returns:
            list[Sample]: List of parsed samples from dataset.
        """
        base_path = Path(__file__).parent / "datasets"
        file_path = base_path / data_set.value

        with open(file_path, "r") as f:
            raw_samples = json.load(f)

        return [
            Sample(sentences=s["text"], segment_lengths=s["segment_length"])
            for s in raw_samples
        ]
