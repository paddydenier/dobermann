from fastapi import FastAPI, HTTPException

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
def segment(algorithm: str, text: list[str]):
    if algorithm not in segmenters:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown algorithm: {algorithm}",
        )

    result = segmenters[algorithm].segment(text)

    return {"segments": result.split(text)}
