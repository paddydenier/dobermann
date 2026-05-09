import re
from pathlib import Path

README_PATH = Path("README.md")

SNIPPETS = {
    "quickstart": Path("examples/graphseg_embeddings.py"),
}


def inject_snippet(content: str, key: str, file_path: Path) -> str:
    code = file_path.read_text().strip()

    replacement = f"<!-- BEGIN:{key} -->\n```python\n{code}\n```\n<!-- END:{key} -->"

    pattern = re.compile(rf"<!-- BEGIN:{key} -->.*?<!-- END:{key} -->", re.DOTALL)

    if not pattern.search(content):
        raise ValueError(f"Markers for '{key}' not found in README")

    return pattern.sub(replacement, content)


def main():
    readme = README_PATH.read_text()

    for key, path in SNIPPETS.items():
        readme = inject_snippet(readme, key, path)

    README_PATH.write_text(readme)
    print("README updated successfully.")


if __name__ == "__main__":
    main()
