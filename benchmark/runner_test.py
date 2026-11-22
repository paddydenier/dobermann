import pytest

from benchmark.loader import DataHandler
from benchmark.runner import DocumentResult, average, run
from benchmark.types import DataSet
from dobermann import SegmentationResult


class SingleSegment:
    """Puts every document into a single segment."""

    def segment(self, sentences):
        return SegmentationResult(
            segment_lengths=[len(sentences)], sentences=sentences, runtime=0.0
        )


class DropsSentence:
    """Returns segment lengths that do not cover all sentences."""

    def segment(self, sentences):
        return SegmentationResult(
            segment_lengths=[len(sentences) - 1], sentences=sentences, runtime=0.0
        )


def test_run_returns_one_result_per_document():
    results = run({"single": SingleSegment()}, [DataSet.CHOI], limit=2)
    samples = DataHandler.samples(DataSet.CHOI)[:2]

    documents = results["single"]["CHOI"]
    assert [d.index for d in documents] == [0, 1]
    for document, sample in zip(documents, samples):
        assert document.reference_lengths == sample.segment_lengths
        assert document.segment_lengths == [len(sample.sentences)]
        assert {"pk_default", "wd_default", "ghd"} <= document.metrics.keys()


def test_run_names_the_failing_document():
    with pytest.raises(ValueError, match="drops on CHOI document 0"):
        run({"drops": DropsSentence()}, [DataSet.CHOI], limit=1)


def test_average():
    documents = [
        DocumentResult(0, [1], [1], {"pk_default": 0.2, "ghd": 2.0}),
        DocumentResult(1, [1], [1], {"pk_default": 0.4, "ghd": 4.0}),
    ]

    assert average(documents) == pytest.approx({"pk_default": 0.3, "ghd": 3.0})
    assert average([]) == {}
