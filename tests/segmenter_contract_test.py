import pytest

from dobermann import (DataHandler, DataSet, GraphSegEmbeddings,
                       TextTilingEmbeddings)


@pytest.fixture(
    params=[
        TextTilingEmbeddings("all-MiniLM-L6-v2"),
        GraphSegEmbeddings("all-MiniLM-L6-v2"),
    ]
)
def segmenter(request):
    return request.param


def test_equivalent_lengths(segmenter):
    # sum of segment lengths equivalent to number of sentences
    sentences = [
        "Cats are animals.",
        "Dogs are animals.",
        "Python is a programming language.",
        "Functions can return values.",
    ]
    result = segmenter.segment(sentences)
    assert sum(result.segment_lengths) == len(sentences)


# TODO: single, zero, two sentences base cases


def test_zero_sentence(segmenter):
    pass


def test_one_sentence(segmenter):
    result = segmenter.segment(["This is a single sentence."])
    assert sum(result.segment_lengths) == 1


def test_two_sentence(segmenter):
    # two obviously unrelated sentences
    pass


def test_segments_deterministic(segmenter):
    # same input consistently produces same output
    sentences = [
        "Cats are animals.",
        "Dogs are animals.",
        "Python is a programming language.",
        "Functions can return values.",
    ]

    result1 = segmenter.segment(sentences)
    result2 = segmenter.segment(sentences)
    assert result1.segment_lengths == result2.segment_lengths


# TODO: add integration testing with datasets
# segmentation must work for all included datasets
