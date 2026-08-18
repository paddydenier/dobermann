# working document example
from dobermann import Document, GraphSegEmbeddings

text = "Cats are domesticated mammals that are commonly kept as pets. They belong to the family Felidae and are known for their agility. Cats have sharp claws and excellent night vision. Many cats communicate using vocalizations such as meowing and purring. Dogs are also domesticated mammals and are among the most common household pets. They belong to the family Canidae and have a strong sense of smell. Dogs have been bred for many different purposes, including hunting and herding. Many dogs are trained to assist humans in various tasks. Python is a high-level programming language used for many different applications. It is widely used in web development, data science, and automation. Python uses indentation to define blocks of code. Functions in Python can accept arguments and return values. A function is defined using the def keyword. Python also provides many built-in data structures such as lists and dictionaries."

document = Document.from_text(text)
sentences = document.sentences

segmenter = GraphSegEmbeddings("all-minilm-l6-v2")
segmentation_result = segmenter.segment(sentences)

chunks = segmentation_result.split(sentences)

print(chunks)
