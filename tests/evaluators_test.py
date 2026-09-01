import pytest

from dobermann import SegmentationEvaluator

pytestmark = pytest.mark.skip(reason="Refactoring")


@pytest.fixture
def evaluator():
    return SegmentationEvaluator()


def test_perfect_segmentation(evaluator):

    ref = [10, 10, 10]
    hyp = [10, 10, 10]

    result = evaluator.evaluate(hyp_len=hyp, ref_len=ref)

    assert result.pk["default"] == 0.0
    assert result.wd["default"] == 0.0
    assert result.ghd == 0.0


def test_sum_mismatch_raises(evaluator):
    ref = [3, 2, 4]
    hyp = [2, 2, 4]  # sum differs

    with pytest.raises(ValueError, match="same total length"):
        evaluator.evaluate(hyp_len=hyp, ref_len=ref)


def test_invalid_segment_lengths(evaluator):
    ref = [3, 0, 4]
    hyp = [2, 3, 2]

    with pytest.raises(ValueError, match="positive integers"):
        evaluator.evaluate(hyp_len=hyp, ref_len=ref)
