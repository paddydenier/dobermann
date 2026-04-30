import pytest

from dobermann import GraphSegEmbeddings, TextTilingEmbeddings


@pytest.fixture(
    params=[
        TextTilingEmbeddings("all-MiniLM-L6-v2"),
        GraphSegEmbeddings("all-MiniLM-L6-v2"),
    ]
)
def segmenter(request):
    return request.param


def test_segment_lengths_cover_all_sentences(segmenter):
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


def test_sentences_list_of_int(segmenter):
    sentences = [i for i in range(10)]
    with pytest.raises(TypeError, match="all elements in sentences must be str"):
        segmenter.segment(sentences)


def test_sentences_mixed_type(segmenter):
    sentences = ["This is a sentence", 1, 2, True]
    with pytest.raises(TypeError, match="all elements in sentences must be str"):
        segmenter.segment(sentences)


def test_sentences_empty_list(segmenter):
    sentences = []
    with pytest.raises(ValueError, match="sentences must be nonempty"):
        segmenter.segment(sentences)


def test_one_sentence(segmenter):
    result = segmenter.segment(["This is a single sentence."])
    assert sum(result.segment_lengths) == 1


# TODO: test_two_sentences


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


# TODO: test_segmenters_support_all_datasets
