import numpy as np
import pytest

from dobermann import DataHandler, DataSet, GraphSegEmbeddings


@pytest.fixture
def segmenter():
    return GraphSegEmbeddings("all-MiniLM-L6-v2")


@pytest.fixture
def sentences():
    return ["A", "B", "C", "D"]


def test_metadata_contains_expected_keys(segmenter, sentences):
    result = segmenter.segment(sentences)

    expected = {
        "embeddings",
        "similarity_matrix",
        "graph",
        "communities",
        "labels",
        "smoothed_labels",
    }

    assert expected <= result.metadata.keys()


def test_similarity_matrix_shape(segmenter, sentences):
    result = segmenter.segment(sentences)
    sim = result.metadata["similarity_matrix"]

    n = len(sentences)

    assert sim.shape == (n, n)


def test_graph_one_node_per_sentence(segmenter, sentences):

    result = segmenter.segment(sentences)
    graph = result.metadata["graph"]

    assert graph.number_of_nodes() == len(sentences)


def test_labels_match_sentence_count(segmenter, sentences):
    result = segmenter.segment(sentences)

    assert len(result.metadata["labels"]) == len(sentences)
    assert len(result.metadata["smoothed_labels"]) == len(sentences)


def test_similarity_matrix_diagonal_is_one(segmenter, sentences):
    result = segmenter.segment(sentences)
    sim = result.metadata["similarity_matrix"]

    assert np.allclose(np.diag(sim), 1.0)


# TODO: test_similarity_matrix_is_symmetri
# TODO: better Metadata consistency check.
