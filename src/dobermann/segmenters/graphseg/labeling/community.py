from .base import Labeler


class CommunityLabeler(Labeler):
    def label(
        self,
        communities: list[list[int]],
        n_sentences: int,
    ) -> list[int]:
        labels = [-1] * n_sentences

        for cid, community in enumerate(communities):
            for idx in community:
                labels[idx] = cid

        # Isolated nodes remain unique singleton labels.
        next_label = len(communities)

        for i in range(n_sentences):
            if labels[i] == -1:
                labels[i] = next_label
                next_label += 1

        return labels
