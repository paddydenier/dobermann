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


def test_sentences_not_list(segmenter):
    sentences = "This is a sentence. This is another sentence."
    with pytest.raises(TypeError, match="sentences must be a list of str"):
        segmenter.segment(sentences)


def test_list_of_int(segmenter):
    sentences = [i for i in range(10)]
    with pytest.raises(TypeError, match="all elements in sentences must be str"):
        segmenter.segment(sentences)


def test_sentences_mixed_type(segmenter):
    sentences = ["This is a sentence", 1, 2, True]
    with pytest.raises(TypeError, match="all elements in sentences must be str"):
        segmenter.segment(sentences)


def test_zero_sentence(segmenter):
    sentences = []
    with pytest.raises(ValueError, match="sentences must be nonempty"):
        segmenter.segment(sentences)


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
