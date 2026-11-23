import subprocess
import sys
from types import ModuleType

import pytest

import dobermann
from dobermann import IdentityPreProcessor, PreProcessor, Segmenter
from dobermann.segmenters import graphseg, texttiling

HEAVY_MODULES = ("torch", "sentence_transformers", "transformers")

PUBLIC_MODULES = [dobermann, texttiling, graphseg]

TOP_LEVEL_API = {
    "Document",
    "TextTiling",
    "GraphSeg",
    "Segmenter",
    "SegmentationResult",
    "SegmentationEvaluator",
    "EvaluationResult",
    "Embedder",
    "SentenceTransformerEmbedder",
    "PreProcessor",
    "IdentityPreProcessor",
}

TEXTTILING_API = {
    "TextTiling",
    "TextTilingEmbeddings",
    "Similarity",
    "CosineSimilarity",
    "SignalSmoother",
    "MovingAverageSmoother",
    "BoundaryDetector",
    "AdaptiveValleyBoundaryDetector",
    "BoundaryPostProcessor",
    "BoundaryToLengthProcessor",
}

GRAPHSEG_API = {
    "GraphSeg",
    "GraphSegEmbeddings",
    "SimilarityMatrix",
    "CosineSimilarityMatrix",
    "GraphBuilder",
    "WeightedGraphBuilder",
    "CommunityDetector",
    "GreedyModularityCommunityDetector",
    "Labeler",
    "CommunityLabeler",
    "LabelSmoother",
    "MajorityVoteSmoother",
    "LabelPostProcessor",
    "LabelsToSegmentsPostProcessor",
}

# The facades are exported by `dobermann`; their algorithm package lists them
# in `__all__` too, as the entry point of the algorithm's own namespace.
FACADES = {"TextTiling", "GraphSeg"}


@pytest.mark.parametrize("module", PUBLIC_MODULES, ids=lambda m: m.__name__)
def test_all_has_no_duplicates(module: ModuleType):
    assert len(module.__all__) == len(set(module.__all__))


@pytest.mark.parametrize("module", PUBLIC_MODULES, ids=lambda m: m.__name__)
def test_all_names_are_attributes(module: ModuleType):
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []


@pytest.mark.parametrize(
    ("module", "expected"),
    [
        (dobermann, TOP_LEVEL_API),
        (texttiling, TEXTTILING_API),
        (graphseg, GRAPHSEG_API),
    ],
    ids=["dobermann", "texttiling", "graphseg"],
)
def test_all_is_pinned(module: ModuleType, expected: set[str]):
    assert set(module.__all__) == expected


def test_segmenters_package_exports_nothing():
    from dobermann import segmenters

    assert segmenters.__all__ == []


def _public_classes(module: ModuleType) -> set[str]:
    return {
        name
        for name, value in vars(module).items()
        if isinstance(value, type) and not name.startswith("_")
    }


def test_every_public_class_has_one_import_path():
    owners: dict[str, list[str]] = {}
    for module in PUBLIC_MODULES:
        for name in _public_classes(module):
            owners.setdefault(name, []).append(module.__name__)

    for facade in FACADES:
        owners[facade].remove(f"dobermann.segmenters.{facade.lower()}")

    duplicated = {name: mods for name, mods in owners.items() if len(mods) > 1}
    assert duplicated == {}


@pytest.mark.parametrize("module", PUBLIC_MODULES, ids=lambda m: m.__name__)
def test_public_classes_are_listed_in_all(module: ModuleType):
    assert _public_classes(module) <= set(module.__all__)


@pytest.mark.parametrize(
    ("module", "base", "implementation"),
    [
        (texttiling, "Similarity", "CosineSimilarity"),
        (texttiling, "SignalSmoother", "MovingAverageSmoother"),
        (texttiling, "BoundaryDetector", "AdaptiveValleyBoundaryDetector"),
        (texttiling, "BoundaryPostProcessor", "BoundaryToLengthProcessor"),
        (graphseg, "SimilarityMatrix", "CosineSimilarityMatrix"),
        (graphseg, "GraphBuilder", "WeightedGraphBuilder"),
        (graphseg, "CommunityDetector", "GreedyModularityCommunityDetector"),
        (graphseg, "Labeler", "CommunityLabeler"),
        (graphseg, "LabelSmoother", "MajorityVoteSmoother"),
        (graphseg, "LabelPostProcessor", "LabelsToSegmentsPostProcessor"),
    ],
    ids=lambda x: getattr(x, "__name__", x).rsplit(".", 1)[-1],
)
def test_stage_implementations_subclass_their_base(
    module: ModuleType, base: str, implementation: str
):
    assert issubclass(getattr(module, implementation), getattr(module, base))


def test_core_bases_are_exported():
    assert issubclass(IdentityPreProcessor, PreProcessor)
    assert issubclass(dobermann.GraphSeg, Segmenter)
    assert issubclass(dobermann.TextTiling, Segmenter)
    assert dobermann.TextTiling is texttiling.TextTiling
    assert dobermann.GraphSeg is graphseg.GraphSeg


def test_import_does_not_load_heavy_modules():
    code = (
        "import sys, dobermann; "
        f"print(','.join(m for m in {HEAVY_MODULES!r} if m in sys.modules))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert result.stdout.strip() == ""
