import pytest

from dobermann import SegmentationEvaluator


def test_perfect_segmentation():

    ref = [10, 10, 10]
    hyp = [10, 10, 10]

    evaluator = SegmentationEvaluator()
    result = evaluator.evaluate(hyp_len=hyp, ref_len=ref)

    assert result.pk["default"] == 0.0
    assert result.wd["default"] == 0.0
    assert result.ghd == 0.0


def test_sum_mismatch_raises():
    ref = [3, 2, 4]
    hyp = [2, 2, 4]  # sum differs

    evaluator = SegmentationEvaluator()

    with pytest.raises(ValueError, match="same total length"):
        result = evaluator.evaluate(hyp_len=hyp, ref_len=ref)


def test_invalid_segment_lengths():
    ref = [3, 0, 4]
    hyp = [2, 3, 2]

    evaluator = SegmentationEvaluator()

    with pytest.raises(ValueError, match="positive integers"):
        result = evaluator.evaluate(hyp_len=hyp, ref_len=ref)
