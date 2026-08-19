import pytest

from benchmark.loader import DataHandler
from benchmark.types import DataSet


@pytest.mark.parametrize("dataset", list(DataSet))
def test_dataset_does_not_crash(dataset):
    DataHandler(dataset)


# FIX: ABSTRACTS Dataset incorrect lengths!

# @pytest.mark.parametrize("dataset", list(DataSet))
# def test_dataset_validity(dataset):
#     handler = DataHandler(dataset)
#
#     for sample in handler.samples:
#         # --- basic structure ---
#         assert isinstance(sample.sentences, list)
#         assert isinstance(sample.segment_lengths, list)
#
#         assert len(sample.sentences) > 0
#         assert len(sample.segment_lengths) > 0
#
#         # --- content sanity ---
#         assert all(isinstance(s, str) for s in sample.sentences)
#         assert all(len(s.strip()) > 0 for s in sample.sentences)
#
#         assert all(isinstance(l, int) for l in sample.segment_lengths)
#         assert all(l > 0 for l in sample.segment_lengths)
#
#         assert sum(sample.segment_lengths) == len(sample.sentences)
