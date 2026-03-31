from dobermann import DataHandler, DataSet


def test_data_loading():
    ds = DataHandler(DataSet.CHOI)

    assert len(ds.samples) > 0

    sample = ds.samples[0]

    assert isinstance(sample.text, list)
    assert isinstance(sample.segment_lengths, list)
