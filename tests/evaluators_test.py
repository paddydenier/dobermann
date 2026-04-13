from dobermann import SegmentationEvaluator


def test_perfect_segmentation():
    ev = SegmentationEvaluator()

    gt = [10, 10, 10]
    pred = [10, 10, 10]

    assert ev.pk(gt, pred) == 0.0
    assert ev.windowdiff(gt, pred) == 0.0


def test_single_boundary_shift():
    ev = SegmentationEvaluator()

    gt = [10, 10, 10]
    pred = [11, 9, 10]

    pk = ev.pk(gt, pred)
    wd = ev.windowdiff(gt, pred)

    assert 0 < pk < 0.5
    assert 0 < wd < 0.5
