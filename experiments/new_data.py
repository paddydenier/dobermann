from dobermann import DataHandler, DataSet

# samples = DataHandler(DataSet.CHOI).samples
# sample = samples[888]
# sentences = sample.sentences

samples = DataHandler.samples(DataSet.CHOI)
sample = samples[888]
sentences = sample.sentences

print(sentences[0])

# segmenter = GraphSegEmbeddings("all-MiniLM-L6-v2")
# segmentation_result = segmenter.segment(sentences)
#
# chunks = segmentation_result.split(sentences)
#
# print(chunks)
