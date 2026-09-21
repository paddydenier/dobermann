from .base import PreProcessor


class IdentityPreProcessor(PreProcessor):
    def process(self, sentences: list[str]) -> list[str]:
        if not isinstance(sentences, list):
            raise TypeError("sentences must be a list of str")

        if not all(isinstance(sentence, str) for sentence in sentences):
            raise TypeError("all elements in sentences must be str")

        if not sentences:
            raise ValueError("sentences must be nonempty")

        return sentences
