import pytest

from dobermann import DataHandler, DataSet, TextTilingEmbeddings

# TODO: test that texttiling runs


@pytest.fixture
def segmenter():
    return TextTilingEmbeddings("all-MiniLM-L6-v2")


def test_correct_lengths(segmenter):
    sentences = [
        "Cats are animals.",
        "Dogs are animals.",
        "Python is a programming language.",
        "Functions can return values.",
    ]
    result = segmenter.segment(sentences)

    # segment_lengths match sentences
    assert sum(result.segment_lengths) == len(sentences)
    # similarities are one less
    assert len(result.metadata["similarities"]) == len(sentences) - 1
    # sentences and embeddings match
    assert result.metadata["embeddings"].shape[0] == len(sentences)


# TODO: add integration testing with datasets


# TODO: add empty sentences
def test_single_sentence(segmenter):
    result = segmenter.segment(["hello"])
    assert sum(result.segment_lengths) == 1


def test_segments_deterministic(segmenter):
    sentences = [
        "Cats are animals.",
        "Dogs are animals.",
        "Python is a programming language.",
        "Functions can return values.",
    ]

    result1 = segmenter.segment(sentences)
    result2 = segmenter.segment(sentences)
    assert result1.segment_lengths == result2.segment_lengths
