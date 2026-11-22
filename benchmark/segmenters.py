from collections.abc import Callable

from dobermann import GraphSeg, TextTiling
from dobermann.segmenters.abstract import Segmenter

SEGMENTERS: dict[str, Callable[[], Segmenter]] = {
    "text_tiling": TextTiling.default,
    "graph_seg": GraphSeg.default,
}
"""Segmenters evaluated by the benchmark, as factories so models load lazily."""
