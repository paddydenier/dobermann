# TODO: add tests for api endpoints
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Dobermann API is running"}


def test_segment_text_tiling():
    mock_segmenter = MagicMock()
    mock_segmenter.segment.return_value = MagicMock(
        split=MagicMock(return_value=["First segment.", "Second segment."])
    )

    with patch.dict(
        "api.main.segmenters",
        {"text_tiling": mock_segmenter},
    ):
        response = client.post(
            "/segment?algorithm=text_tiling",
            json="First sentence. Second sentence.",
        )

    assert response.status_code == 200
    assert response.json() == {"segments": ["First segment.", "Second segment."]}


def test_segment_graph_seg():
    mock_segmenter = MagicMock()
    mock_segmenter.segment.return_value = MagicMock(
        split=MagicMock(return_value=["First segment.", "Second segment."])
    )

    with patch.dict(
        "api.main.segmenters",
        {"graph_seg": mock_segmenter},
    ):
        response = client.post(
            "/segment?algorithm=graph_seg",
            json="First sentence. Second sentence.",
        )

    assert response.status_code == 200
    assert response.json() == {"segments": ["First segment.", "Second segment."]}


def test_segment_unknown_algorithm():
    response = client.post(
        "/segment?algorithm=invalid",
        json="Some text.",
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Unknown algorithm: invalid",
    }


def test_segment_missing_algorithm():
    response = client.post(
        "/segment",
        json="Some text.",
    )

    assert response.status_code == 422


def test_segment_missing_text():
    response = client.post(
        "/segment?algorithm=text_tiling",
    )

    assert response.status_code == 422
