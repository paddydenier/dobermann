"""Response models, so the OpenAPI schema documents every response."""

from typing import Literal

from pydantic import BaseModel


class RootResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    device: str
    """The device the model runs on, such as ``"cpu"`` or ``"cuda:0"``."""


class AlgorithmsResponse(BaseModel):
    algorithms: list[str]


class SegmentResponse(BaseModel):
    segments: list[str]
    """The segments in order, each one its sentences joined by spaces."""


class ErrorResponse(BaseModel):
    detail: str
