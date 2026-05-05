# dobermann

Dobermann is a modern Python library for discourse segmentation, evaluation, and visualization, combining classical computational linguistics literature with contemporary NLP and embedding-based methods.

![ci workflow](https://github.com/paddydenier/dobermann/actions/workflows/ci.yml/badge.svg)

## Features

- ✂️ Discourse Segmentation Algorithms from Computational Linguistics Literature
- 📈 Built-In Datasets and Evaluation Metrics
- 👀 Visualization Tools
- 📖 Free and Open Source

## Installation

Dobermann can conveniently be installed through the pip package manager:

```bash
pip install dobermann
```

Alternatively, clone the repository to access the full range of tools or contribute to the project:

```bash
git clone https://github.com/paddydenier/dobermann.git
cd dobermann
pip install -e .
```

TODO: explain constraints.

## Usage

Minimal segmentation and evaluation workflow example:

<!-- BEGIN:quickstart -->
```python
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
```
<!-- END:quickstart -->

## Contribution and Development Workflows

TODO: explain makefile.
