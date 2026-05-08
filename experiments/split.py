from dobermann import DataHandler, DataSet, GraphSegEmbeddings

samples = DataHandler(DataSet.CHOI).samples
sample = samples[888]
sentences = sample.sentences

segmenter = GraphSegEmbeddings("all-MiniLM-L6-v2")
segmentation_result = segmenter.segment(sentences)

chunks = segmentation_result.split(sentences)

print(chunks)
