from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException, Request

from dobermann import (
    Document,
    Embedder,
    GraphSeg,
    Segmenter,
    SentenceTransformerEmbedder,
    TextTiling,
)


def build_segmenters(embedder: Embedder) -> dict[str, Segmenter]:
    """Build the available segmenters around one shared embedder."""
    return {
        "text_tiling": TextTiling.default(embedder=embedder),
        "graph_seg": GraphSeg.default(embedder=embedder),
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model and segmenters once, before the first request.

    A model that fails to load stops the server at startup instead of
    failing the first request.
    """
    embedder = SentenceTransformerEmbedder.default()
    app.state.segmenters = build_segmenters(embedder)
    yield


app = FastAPI(lifespan=lifespan)


def get_segmenters(request: Request) -> dict[str, Segmenter]:
    """Return the segmenters loaded at startup."""
    return request.app.state.segmenters


@app.get("/")
def root():
    return {"message": "Dobermann API is running"}


@app.post("/segment")
def segment(
    algorithm: str,
    text: Annotated[str, Body()],
    segmenters: Annotated[dict[str, Segmenter], Depends(get_segmenters)],
):
    if algorithm not in segmenters:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm: {algorithm}",
        )

    document = Document.from_text(text)

    if not document.sentences:
        raise HTTPException(
            status_code=400,
            detail="Text contains no sentences",
        )

    result = segmenters[algorithm].segment(document.sentences)

    return {"segments": [" ".join(segment) for segment in result.segments]}
