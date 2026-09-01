from dataclasses import fields
from numbers import Number

from sentence_transformers import SentenceTransformer

from benchmark.loader import DataHandler
from benchmark.types import DataSet
from dobermann import (
    CosineSimilarity,
    GraphSegEmbeddings,
    MovingAverageSmoother,
    SegmentationEvaluator,
    SentenceTransformerEmbedder,
    TextTilingEmbeddings,
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformerEmbedder(model)

smoother = MovingAverageSmoother()
similarity = CosineSimilarity()


def evaluate_segmenter(segmenter, samples, evaluator):
    results = []

    for i, sample in enumerate(samples):
        print(f"Processing sample {i + 1}/{len(samples)}")

        segmentation = segmenter.segment(sample.sentences)

        result = evaluator.evaluate(
            hyp_len=segmentation.segment_lengths,
            ref_len=sample.segment_lengths,
        )

        results.append(result)

    averages = {}

    for field in fields(results[0]):
        metric = field.name
        values = [getattr(result, metric) for result in results]

        if all(isinstance(value, Number) for value in values):
            averages[metric] = sum(values) / len(values)

    return averages


def main():
    samples = DataHandler.samples(DataSet.CHOI)
    evaluator = SegmentationEvaluator()

    segmenters = {
        "TextTilingEmbeddings": TextTilingEmbeddings(embedder, similarity, smoother),
        "GraphSegEmbeddings": GraphSegEmbeddings("all-MiniLM-L6-v2"),
    }

    all_results = {}

    for name, segmenter in segmenters.items():
        print(f"\nRunning {name}...")

        all_results[name] = evaluate_segmenter(
            segmenter,
            samples,
            evaluator,
        )

    print("\n=== Average Results ===")

    for name, averages in all_results.items():
        print(f"\n{name}:")

        for metric, value in averages.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
