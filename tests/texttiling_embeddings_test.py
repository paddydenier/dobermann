import pytest

from dobermann import TextTilingEmbeddings

pytestmark = pytest.mark.skip(reason="Refactoring")


@pytest.fixture
def segmenter():
    return TextTilingEmbeddings("all-MiniLM-L6-v2")


@pytest.fixture
def sentences():
    return ["A", "B", "C", "D"]


def test_sum_embeddings_equal_length_sentences(segmenter, sentences):
    result = segmenter.segment(sentences)
    assert result.metadata["embeddings"].shape[0] == len(sentences)


def test_length_similarities_one_less_than_length_sentences(segmenter, sentences):
    result = segmenter.segment(sentences)
    assert len(result.metadata["similarities"]) == len(sentences) - 1


def test_smoothed_embeddings_match_similarities(segmenter, sentences):
    result = segmenter.segment(sentences)
    assert len(result.metadata["smoothed"]) == len(result.metadata["similarities"])


# TODO: test_boundaries


# TODO: smoothed embeddings same length as original embeddings
