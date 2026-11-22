"""Benchmark command line interface.

python -m benchmark run   [--full]             print average metrics
python -m benchmark save  [--full] [--force]   save results as the local baseline
python -m benchmark check [--full]             compare results with the baseline
"""

import argparse
import sys

from . import snapshot
from .runner import Results, average, run
from .segmenters import SEGMENTERS
from .types import DataSet

QUICK_LIMIT = 3
"""Documents per dataset used unless ``--full`` is given."""

# ABSTRACTS is excluded until its reference segment lengths are fixed: in
# every document they sum to more than the number of sentences.
DATASETS = [dataset for dataset in DataSet if dataset is not DataSet.ABSTRACTS]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmark", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    for name, help in [
        ("run", "print average metrics per segmenter and dataset"),
        ("save", "save results as the local baseline"),
        ("check", "compare results with the local baseline"),
    ]:
        command = commands.add_parser(name, help=help)
        command.add_argument(
            "--full",
            action="store_true",
            help=f"use all documents instead of the first {QUICK_LIMIT} per dataset",
        )
        if name == "save":
            command.add_argument(
                "--force",
                action="store_true",
                help="replace an existing baseline",
            )

    args = parser.parse_args(argv)
    mode = "full" if args.full else "quick"

    if args.command == "check":
        baseline = snapshot.load(mode)
        if baseline is None:
            print(
                f"No {mode} baseline found at {snapshot.path_for(mode)}.\n"
                f"Run `python -m benchmark save{' --full' if args.full else ''}` "
                "before making your change.",
                file=sys.stderr,
            )
            return 2

    if args.command == "save" and not args.force:
        existing = snapshot.load(mode)
        if existing is not None:
            print(
                f"A {mode} baseline already exists at {snapshot.path_for(mode)} "
                f"({snapshot.format_meta(existing.meta)}).\n"
                "Pass --force to replace it.",
                file=sys.stderr,
            )
            return 2

    results = _run(args.full)

    if args.command == "run":
        print(_averages(results))
        return 0

    if args.command == "save":
        path = snapshot.save(results, mode)
        print(f"Saved {mode} baseline to {path}")
        return 0

    print(f"Baseline: {snapshot.format_meta(baseline.meta)}")
    for mismatch in snapshot.version_mismatches(baseline.meta):
        print(f"Warning: {mismatch} since the baseline was saved")
    print()

    comparison = snapshot.compare(baseline.results, results)
    print(snapshot.format_report(comparison))
    print()

    if comparison.has_changes:
        print("Segmentations differ from the baseline.")
        return 1

    print("All segmentations match the baseline.")
    return 0


def _run(full: bool) -> Results:
    segmenters = {name: factory() for name, factory in SEGMENTERS.items()}
    return run(segmenters, DATASETS, limit=None if full else QUICK_LIMIT)


def _averages(results: Results) -> str:
    metrics = snapshot.REPORTED_METRICS
    lines = [
        "  ".join(
            [f"{'segmenter':<12}", f"{'dataset':<14}", f"{'docs':>5}"]
            + [f"{metric:>10}" for metric in metrics]
        )
    ]

    for segmenter, datasets in results.items():
        for dataset, documents in datasets.items():
            averages = average(documents)
            lines.append(
                "  ".join(
                    [f"{segmenter:<12}", f"{dataset:<14}", f"{len(documents):>5}"]
                    + [f"{averages[metric]:>10.4f}" for metric in metrics]
                )
            )

    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
