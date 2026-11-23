import logging
import zlib

import numpy as np
import pytest
import sentence_transformers
import torch
import transformers.utils

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

    def __init__(self, model_name: str, device: str | None = None):
        self.model_name = model_name
        self.requested_device = device
        self.device = device or "cpu"


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


@pytest.mark.parametrize("facade", FACADES, ids=FACADE_IDS)
def test_default_accepts_input_processor_and_embedder(facade):
    embedder = FakeEmbedder()
    input_processor = IdentityPreProcessor()
    segmenter = facade.default(input_processor, embedder=embedder)

    assert segmenter.input_processor is input_processor
    assert segmenter.algorithm.embedder is embedder


@pytest.mark.parametrize("facade", FACADES, ids=FACADE_IDS)
def test_default_accepts_input_processor_by_keyword(facade):
    input_processor = IdentityPreProcessor()
    segmenter = facade.default(input_processor=input_processor, embedder=FakeEmbedder())

    assert segmenter.input_processor is input_processor
    result = segmenter.segment(SENTENCES)
    assert sum(result.segment_lengths) == len(SENTENCES)


@pytest.mark.parametrize("facade", FACADES, ids=FACADE_IDS)
def test_default_input_processor_is_identity(facade):
    segmenter = facade.default(embedder=FakeEmbedder())

    assert isinstance(segmenter.input_processor, IdentityPreProcessor)


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


def test_default_embedder_picks_device_automatically(fake_model):
    embedder = SentenceTransformerEmbedder.default()

    assert embedder._model.requested_device is None


def test_default_embedder_passes_device(fake_model):
    embedder = SentenceTransformerEmbedder.default(device="cpu")

    assert embedder._model.requested_device == "cpu"
    assert embedder.device == "cpu"


def test_default_embedder_rejects_cuda_without_cuda(fake_model, monkeypatch):
    monkeypatch.setattr(transformers.utils, "is_torch_cuda_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA is not available"):
        SentenceTransformerEmbedder.default(device="cuda")


def test_default_embedder_logs_model_and_device(fake_model, caplog):
    with caplog.at_level(logging.INFO, logger="dobermann"):
        SentenceTransformerEmbedder.default(device="cpu")

    assert "sentence-transformers/all-MiniLM-L6-v2" in caplog.text
    assert "cpu" in caplog.text


# Real model: checks that the device reaches sentence-transformers.
def test_default_embedder_on_cpu():
    embedder = SentenceTransformerEmbedder.default(device="cpu")

    assert embedder.device == "cpu"
    assert embedder.embed(SENTENCES).shape[0] == len(SENTENCES)


def test_default_embedder_device_is_auto():
    expected = "cuda:0" if torch.cuda.is_available() else "cpu"

    assert SentenceTransformerEmbedder.default().device == expected


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA is available")
def test_default_embedder_cuda_without_cuda_raises():
    with pytest.raises(ValueError, match="CUDA is not available"):
        SentenceTransformerEmbedder.default(device="cuda")


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
