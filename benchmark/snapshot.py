"""Save benchmark results as a local baseline and compare later runs with it.

Snapshots are meant to be compared on the machine that saved them: they
store exact segment lengths, which can differ across platforms and
library versions.
"""

import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from .runner import DocumentResult, Results, average

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / ".snapshots"

REPORTED_METRICS = ("pk_default", "wd_default", "ghd")

_CELL_WIDTH = 33

_PACKAGES = ("dobermann", "torch", "sentence-transformers", "transformers", "numpy")


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Benchmark results together with where and when they were produced."""

    meta: dict
    results: Results


@dataclass(slots=True)
class Difference:
    """Comparison of one segmenter on one dataset.

    Attributes:
        segmenter: Segmenter name.
        dataset: Dataset name.
        documents: Number of documents compared.
        changed: Indices of documents whose segment lengths changed,
            including documents present on only one side.
        before: Average metrics of the baseline.
        after: Average metrics of the current run.
        status: ``changed``, ``unchanged``, ``new`` or ``missing``.
    """

    segmenter: str
    dataset: str
    documents: int
    changed: list[int] = field(default_factory=list)
    before: dict[str, float] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)
    status: str = "unchanged"


@dataclass(slots=True)
class Comparison:
    """Differences between a baseline and a current run."""

    differences: list[Difference]
    baseline: Results
    current: Results

    @property
    def has_changes(self) -> bool:
        return any(d.status != "unchanged" for d in self.differences)


def path_for(mode: str, directory: Path | None = None) -> Path:
    return (directory or SNAPSHOT_DIR) / f"{mode}.json"


def save(results: Results, mode: str, directory: Path | None = None) -> Path:
    """Write results as the baseline for ``mode`` and return its path."""
    path = path_for(mode, directory)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "meta": _meta(mode),
        "results": {
            segmenter: {
                dataset: [asdict(document) for document in documents]
                for dataset, documents in datasets.items()
            }
            for segmenter, datasets in results.items()
        },
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)

    return path


def load(mode: str, directory: Path | None = None) -> Snapshot | None:
    """Read the baseline for ``mode``, or ``None`` if none was saved."""
    path = path_for(mode, directory)

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = {
        segmenter: {
            dataset: [DocumentResult(**document) for document in documents]
            for dataset, documents in datasets.items()
        }
        for segmenter, datasets in data["results"].items()
    }

    return Snapshot(meta=data["meta"], results=results)


def compare(baseline: Results, current: Results) -> Comparison:
    """Compare the segment lengths and metrics of two runs, document by document."""
    differences = []

    for segmenter in _union(baseline, current):
        before_sets = baseline.get(segmenter, {})
        after_sets = current.get(segmenter, {})

        for dataset in _union(before_sets, after_sets):
            before = before_sets.get(dataset)
            after = after_sets.get(dataset)

            if before is None or after is None:
                documents = after if before is None else before
                differences.append(
                    Difference(
                        segmenter=segmenter,
                        dataset=dataset,
                        documents=len(documents),
                        changed=[d.index for d in documents],
                        before=average(before or []),
                        after=average(after or []),
                        status="new" if before is None else "missing",
                    )
                )
                continue

            before_by_index = {d.index: d for d in before}
            after_by_index = {d.index: d for d in after}
            indices = sorted(before_by_index.keys() | after_by_index.keys())

            changed = [
                i
                for i in indices
                if i not in before_by_index
                or i not in after_by_index
                or before_by_index[i].segment_lengths
                != after_by_index[i].segment_lengths
            ]

            differences.append(
                Difference(
                    segmenter=segmenter,
                    dataset=dataset,
                    documents=len(indices),
                    changed=changed,
                    before=average(before),
                    after=average(after),
                    status="changed" if changed else "unchanged",
                )
            )

    return Comparison(differences=differences, baseline=baseline, current=current)


def format_report(comparison: Comparison, max_documents: int = 5) -> str:
    """Render a comparison as a human-readable table plus the most changed documents."""
    lines = [_header()]

    for d in comparison.differences:
        changed = (
            d.status
            if d.status in ("new", "missing")
            else f"{len(d.changed)}/{d.documents}"
        )
        cells = [f"{d.segmenter:<12}", f"{d.dataset:<14}", f"{changed:>9}"]
        for metric in REPORTED_METRICS:
            cells.append(_metric_cell(d.before.get(metric), d.after.get(metric)))
        lines.append("  ".join(cells))

    documents = _most_changed(comparison, max_documents)
    if documents:
        lines += ["", f"Most changed documents (by |delta {REPORTED_METRICS[0]}|):"]
        for segmenter, dataset, before, after in documents:
            lines += [
                f"  {segmenter} / {dataset} #{after.index}: "
                f"{REPORTED_METRICS[0]} {before.metrics[REPORTED_METRICS[0]]:.4f}"
                f" -> {after.metrics[REPORTED_METRICS[0]]:.4f}",
                f"    ref    {_lengths(after.reference_lengths)}",
                f"    before {_lengths(before.segment_lengths)}",
                f"    after  {_lengths(after.segment_lengths)}",
            ]

    return "\n".join(lines)


def format_meta(meta: dict) -> str:
    commit = meta.get("commit") or "unknown commit"
    dirty = " + uncommitted changes" if meta.get("dirty") else ""
    return f"{commit[:10]}{dirty}, saved {meta.get('created', '?')}"


def version_mismatches(meta: dict) -> list[str]:
    """Describe library versions that differ from the baseline's."""
    current = _versions()
    saved = meta.get("versions", {})
    return [
        f"{package} {saved.get(package)} -> {current.get(package)}"
        for package in _PACKAGES
        if saved.get(package) != current.get(package)
    ]


def _union(a: dict, b: dict) -> list[str]:
    return list(a) + [key for key in b if key not in a]


def _header() -> str:
    cells = [f"{'segmenter':<12}", f"{'dataset':<14}", f"{'changed':>9}"]
    cells += [f"{metric:^{_CELL_WIDTH}}" for metric in REPORTED_METRICS]
    return "  ".join(cells)


def _metric_cell(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        value = after if before is None else before
        return f"{'-' if value is None else f'{value:.4f}':^{_CELL_WIDTH}}"

    delta = after - before
    if delta == 0:
        change = "="
    else:
        change = f"{'+' if delta > 0 else '-'}{abs(delta):.4f}"
    return f"{before:>9.4f} -> {after:>9.4f} {change:>10}"


def _most_changed(comparison: Comparison, limit: int):
    metric = REPORTED_METRICS[0]
    candidates = []

    for d in comparison.differences:
        if d.status != "changed":
            continue
        before = {doc.index: doc for doc in comparison.baseline[d.segmenter][d.dataset]}
        after = {doc.index: doc for doc in comparison.current[d.segmenter][d.dataset]}
        for index in d.changed:
            if index in before and index in after:
                candidates.append((d.segmenter, d.dataset, before[index], after[index]))

    candidates.sort(
        key=lambda c: abs(c[3].metrics[metric] - c[2].metrics[metric]), reverse=True
    )
    return candidates[:limit]


def _lengths(lengths: list[int], limit: int = 20) -> str:
    if len(lengths) <= limit:
        return str(lengths)
    shown = ", ".join(map(str, lengths[:limit]))
    return f"[{shown}, ... {len(lengths) - limit} more]"


def _meta(mode: str) -> dict:
    return {
        "mode": mode,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": _versions(),
    }


def _versions() -> dict[str, str | None]:
    versions = {}
    for package in _PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()
