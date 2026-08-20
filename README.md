# dobermann

Dobermann is a modern Python library for discourse segmentation, evaluation, and visualization, combining classical computational linguistics literature with contemporary NLP and embedding-based methods.

![ci workflow](https://github.com/paddydenier/dobermann/actions/workflows/ci.yml/badge.svg)
![docs workflow](https://github.com/paddydenier/dobermann/actions/workflows/docs.yml/badge.svg)

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

## Usage

Minimal segmentation and evaluation workflow example:

<!-- BEGIN:quickstart -->

```python
from dobermann import Document, GraphSegEmbeddings, SegmentationEvaluator

text = "Cats are domesticated mammals that are commonly kept as pets. They belong to the family Felidae and are known for their agility. Cats have sharp claws and excellent night vision. Many cats communicate using vocalizations such as meowing and purring. Dogs are also domesticated mammals and are among the most common household pets. They belong to the family Canidae and have a strong sense of smell. Dogs have been bred for many different purposes, including hunting and herding. Many dogs are trained to assist humans in various tasks. Python is a high-level programming language used for many different applications. It is widely used in web development, data science, and automation. Python uses indentation to define blocks of code. Functions in Python can accept arguments and return values. A function is defined using the def keyword. Python also provides many built-in data structures such as lists and dictionaries."
document = Document.from_text(text)

segmenter = GraphSegEmbeddings("all-MiniLM-L6-v2")
segmentation_result = segmenter.segment(document.sentences)

print(segmentation_result.split(document.sentences))
```

<!-- END:quickstart -->

## API Usage

For integration into existing pipelines, Dobermann provides a standardized FastAPI interface for exposing its segmentation capabilities through a REST API.

### Starting the API

On default port 8000:

```bash
make backend
```

On custom port, e.g., 8081:

```bash
make backend PORT=8081
```

## Contribution and Development Workflows
