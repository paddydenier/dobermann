import zlib

import numpy as np
import pytest
import sentence_transformers

from api.main import get_segmenters
from dobermann import (
    Embedder,
    GraphSeg,
    IdentityPreProcessor,
    SentenceTransformerEmbedder,
    TextTiling,
)

SENTENCES = [
    "Cats are animals.",
    "Dogs are animals.",
    "Python is a programming language.",
    "Functions can return values.",
]


class FakeEmbedder(Embedder):
    """Returns a deterministic random vector per sentence, without a model."""

    def embed(self, sentences: list[str]) -> np.ndarray:
        return np.array(
            [
                np.random.default_rng(zlib.crc32(s.encode("utf-8"))).random(8)
                for s in sentences
            ]
        )


class FakeSentenceTransformer:
    """Stands in for the real model so that default() loads nothing."""

    def __init__(self, model_name: str):
        self.model_name = model_name


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", FakeSentenceTransformer
    )


FACADES = [TextTiling, GraphSeg]
FACADE_IDS = ["TextTiling", "GraphSeg"]


@pytest.mark.parametrize("facade", FACADES, ids=FACADE_IDS)
def test_default_uses_injected_embedder(facade):
    embedder = FakeEmbedder()
    segmenter = facade.default(embedder=embedder)

    assert segmenter.algorithm.embedder is embedder
    result = segmenter.segment(SENTENCES)
    assert sum(result.segment_lengths) == len(SENTENCES)


def test_texttiling_default_accepts_input_processor_and_embedder():
    embedder = FakeEmbedder()
    input_processor = IdentityPreProcessor()
    segmenter = TextTiling.default(input_processor, embedder=embedder)

    assert segmenter.input_processor is input_processor
    assert segmenter.algorithm.embedder is embedder


def test_segmenters_share_one_embedder():
    embedder = FakeEmbedder()
    text_tiling = TextTiling.default(embedder=embedder)
    graph_seg = GraphSeg.default(embedder=embedder)

    assert text_tiling.algorithm.embedder is graph_seg.algorithm.embedder
    for segmenter in (text_tiling, graph_seg):
        result = segmenter.segment(SENTENCES)
        assert sum(result.segment_lengths) == len(SENTENCES)


def test_default_embedder_is_a_new_model_per_call(fake_model):
    first = SentenceTransformerEmbedder.default()
    second = SentenceTransformerEmbedder.default()

    assert first is not second
    assert first._model is not second._model
    assert first._model.model_name == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.parametrize("facade", FACADES, ids=FACADE_IDS)
def test_default_without_embedder_builds_its_own(fake_model, facade):
    first = facade.default()
    second = facade.default()

    assert isinstance(first.algorithm.embedder, SentenceTransformerEmbedder)
    assert first.algorithm.embedder is not second.algorithm.embedder
    assert first.algorithm.embedder._model is not second.algorithm.embedder._model


def test_api_segmenters_share_one_embedder():
    segmenters = get_segmenters()

    assert (
        segmenters["text_tiling"].algorithm.embedder
        is segmenters["graph_seg"].algorithm.embedder
    )
