from benchmark.loader import DataHandler
from benchmark.types import DataSet
from dobermann import GraphSegEmbeddings, SegmentationEvaluator

# TODO: gen. avg. results for all combinations


def main():
    dataset = DataHandler(DataSet.CHOI)
    sample = dataset.samples[888]

    segmenter = GraphSegEmbeddings("all-MiniLM-L6-v2")
    segmentation = segmenter.segment(sample.sentences)

    evaluator = SegmentationEvaluator()
    result = evaluator.evaluate(
        hyp_len=segmentation.segment_lengths,
        ref_len=sample.segment_lengths,
    )

    print(result)


if __name__ == "__main__":
    main()
