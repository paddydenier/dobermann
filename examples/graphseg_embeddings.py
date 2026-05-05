from dobermann import (DataHandler, DataSet, SegmentationEvaluator,
                       GraphSegEmbeddings)

samples = DataHandler(DataSet.CHOI).samples
sample = samples[888]
sentences = sample.sentences

# accept list of sentences, return list of segment length
segmenter = GraphSegEmbeddings("all-MiniLM-L6-v2")
segmentation_result = segmenter.segment(sentences)

# evaluation
evaluator = SegmentationEvaluator()
eval_result = evaluator.evaluate(hyp_len=segmentation_result.segment_lengths, ref_len=sample.segment_lengths)

print(eval_result)
