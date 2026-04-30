from dobermann import (DataHandler, DataSet, SegmentationEvaluator,
                       GraphSegEmbeddings)

samples = DataHandler(DataSet.CHOI).samples
sample = samples[888]
sentences = sample.sentences

# accept list of sentences, return list of segment length
tt_all_mini = GraphSegEmbeddings("all-MiniLM-L6-v2")
result = tt_all_mini.segment(sentences)
# print(result.metadata["communities"])
print(result.segment_lengths)

evaluator = SegmentationEvaluator()
eval_result = evaluator.evaluate(hyp_len=result.segment_lengths, ref_len=sample.segment_lengths)

print(eval_result)
