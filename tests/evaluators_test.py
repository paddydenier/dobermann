import pytest

from dobermann import SegmentationEvaluator


def test_perfect_segmentation():

    ref = [10, 10, 10]
    hyp = [10, 10, 10]

    evaluator = SegmentationEvaluator(hyp_len=hyp, ref_len=ref)

    assert evaluator.metrics["pk"]["default"] == 0.0
    assert evaluator.metrics["wd"]["default"] == 0.0
    assert evaluator.metrics["ghd"] == 0.0


# TODO: add data validation error tests
def test_sum_mismatch_raises():
    ref = [3, 2, 4]
    hyp = [2, 2, 4]  # sum differs

    with pytest.raises(ValueError, match="same total length"):
        SegmentationEvaluator(ref, hyp)

def test_invalid_segment_lengths():
    ref = [3, 0, 4]
    hyp = [2, 3, 2]

    with pytest.raises(ValueError, match="positive integers"):
        SegmentationEvaluator(ref, hyp)


