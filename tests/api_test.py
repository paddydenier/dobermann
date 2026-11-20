import pytest
from fastapi.testclient import TestClient

from api.main import app, get_segmenters
from dobermann import SegmentationResult

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
    assert response.json() == {"detail": "Unknown algorithm: invalid"}


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


# Integration: real segmenters, so changes to the library API break these.
@pytest.mark.parametrize("algorithm", ["text_tiling", "graph_seg"])
def test_segment_with_default_segmenters(algorithm):
    response = client.post(f"/segment?algorithm={algorithm}", json=TEXT)

    assert response.status_code == 200
    segments = response.json()["segments"]
    assert segments
    assert all(isinstance(segment, str) for segment in segments)
    assert " ".join(segments) == " ".join(SENTENCES)
