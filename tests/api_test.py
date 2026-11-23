import zlib

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from api.config import Settings
from api.main import app, get_embedder, get_segmenters
from dobermann import Embedder, SegmentationResult, SentenceTransformerEmbedder

# Without a `with` block, TestClient skips the lifespan, so no model is
# loaded. These tests override get_segmenters instead.
client = TestClient(app)

TEXT = (
    "Cats are animals. Dogs are animals too. "
    "Python is a programming language. Functions can return values."
)
SENTENCES = [
    "Cats are animals.",
    "Dogs are animals too.",
    "Python is a programming language.",
    "Functions can return values.",
]


class FakeSegmenter:
    """Splits the input into two segments of two sentences each."""

    def __init__(self):
        self.calls = []

    def segment(self, sentences):
        self.calls.append(sentences)
        return SegmentationResult(
            segment_lengths=[2, len(sentences) - 2],
            sentences=sentences,
            runtime=0.0,
        )


class FakeEmbedder(Embedder):
    """Returns a deterministic random vector per sentence, without a model."""

    device = "fake"

    def embed(self, sentences: list[str]) -> np.ndarray:
        return np.array(
            [
                np.random.default_rng(zlib.crc32(s.encode("utf-8"))).random(8)
                for s in sentences
            ]
        )


@pytest.fixture
def fake_default_embedder(monkeypatch):
    """Replaces SentenceTransformerEmbedder.default() and records its calls."""
    embedder = FakeEmbedder()
    calls = []

    def default(device=None):
        calls.append(device)
        return embedder

    monkeypatch.setattr(SentenceTransformerEmbedder, "default", staticmethod(default))
    return embedder, calls


@pytest.fixture
def real_client():
    """A client that runs the lifespan, so the real model is loaded."""
    with TestClient(app) as real_client:
        yield real_client


@pytest.fixture
def fake_segmenter():
    segmenter = FakeSegmenter()
    app.dependency_overrides[get_segmenters] = lambda: {
        "text_tiling": segmenter,
        "graph_seg": segmenter,
    }
    yield segmenter
    app.dependency_overrides.clear()


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Dobermann API is running"}


@pytest.mark.parametrize("algorithm", ["text_tiling", "graph_seg"])
def test_segment(fake_segmenter, algorithm):
    response = client.post(f"/segment?algorithm={algorithm}", json=TEXT)

    assert response.status_code == 200
    assert fake_segmenter.calls == [SENTENCES]
    assert response.json() == {
        "segments": [
            "Cats are animals. Dogs are animals too.",
            "Python is a programming language. Functions can return values.",
        ]
    }


def test_segment_unknown_algorithm(fake_segmenter):
    response = client.post("/segment?algorithm=invalid", json=TEXT)

    assert response.status_code == 400
    assert response.json() == {
        "detail": (
            "Unknown algorithm: invalid. Available algorithms: text_tiling, graph_seg"
        )
    }
    assert fake_segmenter.calls == []


def test_health():
    app.dependency_overrides[get_embedder] = FakeEmbedder
    try:
        response = client.get("/health")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "device": "fake"}


def test_algorithms(fake_segmenter):
    response = client.get("/algorithms")

    assert response.status_code == 200
    assert response.json() == {"algorithms": ["text_tiling", "graph_seg"]}


@pytest.mark.parametrize("text", ["", "   "])
def test_segment_empty_text(fake_segmenter, text):
    response = client.post("/segment?algorithm=text_tiling", json=text)

    assert response.status_code == 400
    assert response.json() == {"detail": "Text contains no sentences"}
    assert fake_segmenter.calls == []


def test_segment_missing_algorithm(fake_segmenter):
    response = client.post("/segment", json=TEXT)

    assert response.status_code == 422


def test_segment_missing_text(fake_segmenter):
    response = client.post("/segment?algorithm=text_tiling")

    assert response.status_code == 422


def test_segmenters_are_loaded_at_startup(fake_default_embedder, monkeypatch):
    monkeypatch.delenv("DOBERMANN_DEVICE", raising=False)
    embedder, calls = fake_default_embedder

    with TestClient(app) as startup_client:
        assert calls == [None]
        segmenters = app.state.segmenters
        assert set(segmenters) == {"text_tiling", "graph_seg"}
        for segmenter in segmenters.values():
            assert segmenter.algorithm.embedder is embedder

        response = startup_client.post("/segment?algorithm=graph_seg", json=TEXT)
        health = startup_client.get("/health")

    assert response.status_code == 200
    assert health.json() == {"status": "ok", "device": "fake"}
    assert calls == [None]


@pytest.mark.parametrize(
    ("value", "device"),
    [("auto", None), ("cpu", "cpu"), ("cuda", "cuda"), (" CPU ", "cpu")],
)
def test_settings_device(value, device):
    assert Settings.from_env({"DOBERMANN_DEVICE": value}).device == device


def test_settings_device_defaults_to_auto():
    assert Settings.from_env({}).device is None


def test_settings_reject_invalid_device():
    with pytest.raises(ValueError, match="Invalid DOBERMANN_DEVICE='bogus'"):
        Settings.from_env({"DOBERMANN_DEVICE": "bogus"})


def test_startup_uses_configured_device(fake_default_embedder, monkeypatch):
    monkeypatch.setenv("DOBERMANN_DEVICE", "cpu")
    _, calls = fake_default_embedder

    with TestClient(app):
        assert calls == ["cpu"]


def test_startup_fails_on_invalid_device(fake_default_embedder, monkeypatch):
    monkeypatch.setenv("DOBERMANN_DEVICE", "bogus")
    _, calls = fake_default_embedder

    with pytest.raises(ValueError, match="expected one of auto, cuda, cpu"):
        with TestClient(app):
            pass
    assert calls == []


# Integration: real segmenters, so changes to the library API break these.
@pytest.mark.parametrize("algorithm", ["text_tiling", "graph_seg"])
def test_segment_with_default_segmenters(real_client, algorithm):
    response = real_client.post(f"/segment?algorithm={algorithm}", json=TEXT)

    assert response.status_code == 200
    segments = response.json()["segments"]
    assert segments
    assert all(isinstance(segment, str) for segment in segments)
    assert " ".join(segments) == " ".join(SENTENCES)


def test_health_with_default_embedder(real_client):
    response = real_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    }


def test_algorithms_with_default_segmenters(real_client):
    response = real_client.get("/algorithms")

    assert response.json() == {"algorithms": ["text_tiling", "graph_seg"]}
