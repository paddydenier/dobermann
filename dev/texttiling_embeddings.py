from dobermann import (DataHandler, DataSet, SegmentationEvaluator,
                       TextTilingEmbeddings)

# samples = DataHandler(DataSet.CHOI).samples
# sample = samples[888]
# sentences = sample.sentences

sentences = [123,12345,21312]

# accept list of sentences, return list of segment length
tt_all_mini = TextTilingEmbeddings("all-MiniLM-L6-v2")
segmentation_result = tt_all_mini.segment(sentences)
print(segmentation_result.segment_lengths)
