from typing import Annotated

from fastapi import Body, FastAPI, HTTPException

from dobermann.data import Document
from dobermann.segmenters import GraphSegEmbeddings, TextTilingEmbeddings

app = FastAPI()


segmenters = {
    "text_tiling": TextTilingEmbeddings("all-MiniLM-L6-v2"),
    "graph_seg": GraphSegEmbeddings("all-MiniLM-L6-v2"),
}


@app.get("/")
def root():
    return {"message": "Dobermann API is running"}


@app.post("/segment")
def segment(
    algorithm: str,
    text: Annotated[str, Body()],
):
    if algorithm not in segmenters:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm: {algorithm}",
        )

    document = Document.from_text(text)

    result = segmenters[algorithm].segment(document.sentences)

    return {"segments": result.split(document.sentences)}
