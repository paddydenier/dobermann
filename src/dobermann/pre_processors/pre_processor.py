# TODO: can lemmatize, to lowercase, stopword removal, punctuation removal


from dobermann.data.types import Sample


class PreProcessor:
    # state configuration(e.g. lemmatization, stopword removal, etc.) can be added here
    def __init__(self, lowercase: bool, lemmatize: bool):
        self.lowercase = lowercase
        self.lemmatize = lemmatize

    def process(self, samples: list[Sample]) -> list[Sample]:
        if self.lowercase:
            for sample in samples:
                sample.sentences = [sentence.lower() for sentence in sample.sentences]
        return samples
