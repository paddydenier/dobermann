# TODO: repurpose for TextTilingEmbeddings
import pytest

from dobermann import DataHandler, DataSet, TextTilingEmbeddings

# FOCUS ON MAKING TESTS ON METADATA fields

@pytest.fixture
def segmenter():
    return TextTilingEmbeddings("all-MiniLM-L6-v2")


# TODO: consecutive similarities one less than num of sentences
# TODO: number of embeddings matches number of length of sentences
# TODO: smoothed embeddings same length as original embeddings


def test_correct_lengths(segmenter):
    sentences = [
        "Cats are animals.",
        "Dogs are animals.",
        "Python is a programming language.",
        "Functions can return values.",
    ]
    result = segmenter.segment(sentences)

    # similarities are one less
    assert len(result.metadata["similarities"]) == len(sentences) - 1
    # sentences and embeddings match
    assert result.metadata["embeddings"].shape[0] == len(sentences)
