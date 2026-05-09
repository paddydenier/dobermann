from nltk.tokenize import TextTilingTokenizer
from dobermann import DataHandler, DataSet

samples = DataHandler(DataSet.CHOI).samples
sample = samples[888]
sentences = sample.sentences

tt = TextTilingTokenizer(demo_mode=True)

text = "\n\n".join(sentences)

gap_scores, smooth_scores, depth_scores, boundaries = tt.tokenize(text)


# --- FIX: map safely using min-length guard ---
def boundaries_to_sentence_segments(sentences, boundaries):
    segments = []
    start = 0

    for i, b in enumerate(boundaries):
        if b == 1:
            # clamp to sentence length (CRITICAL FIX)
            end = min(i + 1, len(sentences))
            if start < len(sentences):
                segments.append(sentences[start:end])
            start = end

    # append remaining sentences
    if start < len(sentences):
        segments.append(sentences[start:])

    return segments


segments = boundaries_to_sentence_segments(sentences, boundaries)

# --- Debug ---
print("Sentences:", len(sentences))
print("Boundaries (count):", sum(boundaries))
print("Boundary positions:", [i for i, b in enumerate(boundaries) if b == 1])
print("Segments found:", len(segments))

print("\nSegment sizes:")
print([len(s) for s in segments])
