from .base import PreProcessor


class IdentityPreProcessor(PreProcessor):
    def process(self, sentences: list[str]) -> list[str]:
        return sentences
