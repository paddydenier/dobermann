from functools import cache
from typing import Annotated

from fastapi import Body, Depends, FastAPI, HTTPException

from dobermann import (
    Document,
    GraphSeg,
    Segmenter,
    SentenceTransformerEmbedder,
    TextTiling,
)

app = FastAPI()


@cache
def get_segmenters() -> dict[str, Segmenter]:
    """Build the available segmenters once, on first use.

    Both segmenters share one embedder, so the model is loaded only once.
    """
    embedder = SentenceTransformerEmbedder.default()
    return {
        "text_tiling": TextTiling.default(embedder=embedder),
        "graph_seg": GraphSeg.default(embedder=embedder),
    }


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
