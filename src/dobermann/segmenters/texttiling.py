from collections import Counter

import numpy as np
import spacy


class TextTiling:
    def __init__(
        self, pseudo_sentence_size: int, window_size: int, smoothing_window: int
    ):
        self.nlp = spacy.load("en_core_web_sm")
        self.pseudo_sentence_size = pseudo_sentence_size
        self.window_size = window_size
        self.smoothing_window = smoothing_window

    def segment(self, sentences: list[str]) -> list[int]:
        # --- preprocess ---
        tokens, token_to_sentence, token_is_stop = self._preprocess(sentences)

        # --- build pseudo-sentences (token sequences) ---
        sequences = self._build_sequences(tokens)

        # --- vectorize (IGNORE stopwords here only) ---
        vectors = self._transform(sequences, token_is_stop)

        # --- similarity ---
        sims = self._similarity(vectors)
        sims = self._smooth(sims)

        # --- boundaries (in sequence space) ---
        boundaries = self._boundaries(sims)

        # --- project to sentence boundaries ---
        sentence_boundaries = self._project_boundaries(
            boundaries,
            token_to_sentence,
            len(tokens),
        )

        # --- build sentence segments ---
        segments = self._to_sentence_segments(sentence_boundaries, len(sentences))

        # --- lengths ---
        lengths = [end - start + 1 for start, end in segments]

        return lengths

    def _preprocess(self, sentences):
        docs = list(self.nlp.pipe(sentences))

        tokens = []
        token_to_sentence = []
        token_is_stop = []

        for i, doc in enumerate(docs):
            for token in doc:
                if token.is_punct or token.is_space:
                    continue

                tokens.append(token.lemma_.lower())
                token_to_sentence.append(i)
                token_is_stop.append(token.is_stop)

        return tokens, np.array(token_to_sentence), np.array(token_is_stop)

    def _build_sequences(self, tokens):
        w = self.pseudo_sentence_size
        return [tokens[i : i + w] for i in range(0, len(tokens), w)]

    def _transform(self, sequences, token_is_stop):
        vocab = sorted(set(t for seq in sequences for t in seq))
        vocab_index = {w: i for i, w in enumerate(vocab)}

        vectors = []
        idx = 0  # global token index

        for seq in sequences:
            vec = np.zeros(len(vocab))
            counts = Counter()

            for token in seq:
                if not token_is_stop[idx]:  # 🔥 ignore stopwords ONLY here
                    counts[token] += 1
                idx += 1

            for word, count in counts.items():
                vec[vocab_index[word]] = count

            vectors.append(vec)

        return np.array(vectors)

    def _similarity(self, vectors):
        k = self.window_size
        n = len(vectors)
        sims = []

        for i in range(1, n):
            left = np.sum(vectors[max(0, i - k) : i], axis=0)
            right = np.sum(vectors[i : min(n, i + k)], axis=0)

            denom = np.linalg.norm(left) * np.linalg.norm(right)
            sim = np.dot(left, right) / denom if denom != 0 else 0.0
            sims.append(sim)

        return np.array(sims)

    def _smooth(self, arr):
        window = self.smoothing_window
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode="same")

    def _boundaries(self, sims, min_gap=5, mode="HC"):
        n = len(sims)
        depth = np.zeros(n)

        for i in range(n):
            lmax = sims[i]
            for j in range(i - 1, -1, -1):
                if sims[j] > lmax:
                    lmax = sims[j]
                else:
                    break

            rmax = sims[i]
            for j in range(i + 1, n):
                if sims[j] > rmax:
                    rmax = sims[j]
                else:
                    break

            depth[i] = (lmax - sims[i]) + (rmax - sims[i])

        depth = self._smooth(depth)

        minima = [
            i
            for i in range(1, n - 1)
            if sims[i] < sims[i - 1] and sims[i] < sims[i + 1]
        ]

        mean, std = np.mean(depth), np.std(depth)
        threshold = mean if mode == "LC" else mean + std

        candidates = [(i, depth[i]) for i in minima if depth[i] > threshold]
        candidates.sort(key=lambda x: x[1], reverse=True)

        boundaries = []
        for i, _ in candidates:
            if all(abs(i - b) >= min_gap for b in boundaries):
                boundaries.append(i)

        return sorted(boundaries)

    def _project_boundaries(self, boundaries, token_to_sentence, n_tokens):
        w = self.pseudo_sentence_size

        token_boundaries = [min(b * w, n_tokens - 1) for b in boundaries]

        sentence_boundaries = sorted(
            set(token_to_sentence[tb] for tb in token_boundaries)
        )

        return sentence_boundaries

    def _to_sentence_segments(self, boundaries, n_sentences):
        points = [0] + boundaries + [n_sentences]

        segments = []
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1] - 1
            segments.append((start, end))

        return segments
