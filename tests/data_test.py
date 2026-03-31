from dobermann import DataHandler, DataSet


def test_data_loading():
    ds = DataHandler(DataSet.CHOI)

    assert len(ds.samples) > 0

    sample = ds.samples[0]

    # Sentences are list of strings
    assert isinstance(sample.sentences, list)
    assert all(isinstance(s, str) for s in sample.sentences)

    assert isinstance(sample.segment_lengths, list)
