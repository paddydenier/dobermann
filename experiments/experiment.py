from nltk.tokenize import TextTilingTokenizer

from dobermann import DataHandler, DataSet

samples = DataHandler(DataSet.CHOI).samples
sample = samples[888]
sentences = sample.sentences

tt = TextTilingTokenizer(demo_mode=True)
text = "\n\n".join(sentences)
gap_scores, smooth_scores, depth_scores, boundaries = tt.tokenize(text)
# segments = tt.tokenize(text)

# print(segments)
print(len(boundaries))
print(len(sentences))


# TextTiling paper defaults
# segmenter = TextTiling(pseudo_sentence_size=20, window_size=10, smoothing_window=2)
#
# result = segmenter.segment(sentences)
# print("RES: ", result, sum(result))
# print("GRT: ", sample.segment_lengths, sum(sample.segment_lengths))
#
# evaluator = SegmentationEvaluator(hyp_len=result, ref_len=sample.segment_lengths)

# print(evaluator.hyp_str)
# print(evaluator.ref_str)
# print(evaluator.k_values.values())
# print(evaluator.pkn)
# print(evaluator.pkd)
#
# print(evaluator.metrics["pk"])
# print(evaluator.metrics["wd"])
# print(evaluator.metrics["ghd"])

# TODO: implement standard NLTK texttiling as baseline

# TODO: add helper to visualize sample based on sentences and lengths
