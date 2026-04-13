import pytest

from dobermann import DataHandler, DataSet, TextTiling


# def test_texttiling_segmentations_align():
#     # the number of sentences must match the sum of the ground truth which must match the sum of the result
#     samples = DataHandler(DataSet.WIKI_1024).samples
#     sample = samples[0]
# 
#     segmenter = TextTiling(pseudo_sentence_size=5, window_size=3)
# 
#     result = segmenter.segment(sample.sentences)
# 
#     assert sum(result) == len(
#         sample.sentences
#     ), f"Mismatch: pred_total={sum(result)}, num_sentences={len(sample.sentences)}"
