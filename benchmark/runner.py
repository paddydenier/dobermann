import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from dobermann import EvaluationResult, SegmentationEvaluator
from dobermann.segmenters.abstract import Segmenter

from .loader import DataHandler
from .types import DataSet


@dataclass(frozen=True, slots=True)
class DocumentResult:
    """Segmentation and evaluation of a single dataset document.

    Attributes:
        index: Position of the document in its dataset.
        reference_lengths: Ground truth segment lengths.
        segment_lengths: Segment lengths predicted by the segmenter.
        metrics: Flat evaluation metrics, e.g. ``pk_default`` or ``ghd``.
    """

    index: int
    reference_lengths: list[int]
    segment_lengths: list[int]
    metrics: dict[str, float]


Results = dict[str, dict[str, list[DocumentResult]]]
"""Document results keyed by segmenter name, then dataset name."""


def run(
    segmenters: Mapping[str, Segmenter],
    datasets: Iterable[DataSet],
    limit: int | None = None,
    evaluator: SegmentationEvaluator | None = None,
) -> Results:
    """Segment and evaluate the documents of each dataset with each segmenter.

    Args:
        segmenters: Segmenters to evaluate, keyed by name.
        datasets: Datasets to evaluate on.
        limit: Only use the first ``limit`` documents of each dataset.
        evaluator: Evaluator used to score the segmentations.

    Returns:
        Results: Document results keyed by segmenter name, then dataset name.
    """
    evaluator = evaluator or SegmentationEvaluator()
    samples = {
        dataset.name: DataHandler.samples(dataset)[:limit] for dataset in datasets
    }

    results: Results = {}

    for name, segmenter in segmenters.items():
        results[name] = {}

        for dataset, dataset_samples in samples.items():
            print(
                f"{name} on {dataset} ({len(dataset_samples)} documents)",
                file=sys.stderr,
            )
            documents = []

            for index, sample in enumerate(dataset_samples):
                try:
                    documents.append(_evaluate(index, sample, segmenter, evaluator))
                except ValueError as error:
                    raise ValueError(
                        f"{name} on {dataset} document {index}: {error}"
                    ) from error

            results[name][dataset] = documents

    return results


def average(documents: list[DocumentResult]) -> dict[str, float]:
    """Average each metric over the given documents."""
    if not documents:
        return {}

    return {
        metric: sum(document.metrics[metric] for document in documents) / len(documents)
        for metric in documents[0].metrics
    }


def _evaluate(index, sample, segmenter, evaluator) -> DocumentResult:
    segmentation = segmenter.segment(sample.sentences)
    evaluation = evaluator.evaluate(
        ref_len=sample.segment_lengths,
        hyp_len=segmentation.segment_lengths,
    )

    return DocumentResult(
        index=index,
        reference_lengths=list(sample.segment_lengths),
        segment_lengths=list(segmentation.segment_lengths),
        metrics=_flatten(evaluation),
    )


def _flatten(evaluation: EvaluationResult) -> dict[str, float]:
    metrics = {f"pk_{k}": float(v) for k, v in evaluation.pk.items()}
    metrics |= {f"wd_{k}": float(v) for k, v in evaluation.wd.items()}
    metrics["ghd"] = float(evaluation.ghd)
    return metrics
