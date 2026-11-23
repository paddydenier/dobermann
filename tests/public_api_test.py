import subprocess
import sys
from types import ModuleType

import pytest

import dobermann
from dobermann import IdentityPreProcessor, PreProcessor, Segmenter
from dobermann.segmenters import graphseg, texttiling
from dobermann.segmenters.graphseg import (
    CommunityDetector,
    CommunityLabeler,
    GreedyModularityCommunityDetector,
    Labeler,
    LabelPostProcessor,
    LabelSmoother,
    LabelsToSegmentsPostProcessor,
    MajorityVoteSmoother,
)

HEAVY_MODULES = ("torch", "sentence_transformers", "transformers")


@pytest.mark.parametrize(
    "module", [dobermann, graphseg, texttiling], ids=lambda m: m.__name__
)
def test_all_has_no_duplicates(module: ModuleType):
    assert len(module.__all__) == len(set(module.__all__))


@pytest.mark.parametrize(
    "module", [dobermann, graphseg, texttiling], ids=lambda m: m.__name__
)
def test_all_names_are_attributes(module: ModuleType):
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert missing == []


def test_core_bases_are_exported():
    assert {"Segmenter", "PreProcessor", "IdentityPreProcessor"} <= set(
        dobermann.__all__
    )
    assert issubclass(IdentityPreProcessor, PreProcessor)
    assert issubclass(dobermann.GraphSeg, Segmenter)
    assert issubclass(dobermann.TextTiling, Segmenter)


@pytest.mark.parametrize(
    ("base", "implementation"),
    [
        (CommunityDetector, GreedyModularityCommunityDetector),
        (Labeler, CommunityLabeler),
        (LabelSmoother, MajorityVoteSmoother),
        (LabelPostProcessor, LabelsToSegmentsPostProcessor),
    ],
    ids=lambda cls: cls.__name__,
)
def test_graphseg_stages_are_exported(base: type, implementation: type):
    assert base.__name__ in graphseg.__all__
    assert implementation.__name__ in graphseg.__all__
    assert issubclass(implementation, base)


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
