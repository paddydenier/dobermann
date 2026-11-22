from benchmark.runner import average, run
from benchmark.segmenters import SEGMENTERS
from benchmark.types import DataSet


def main():
    segmenters = {name: factory() for name, factory in SEGMENTERS.items()}
    results = run(segmenters, [DataSet.CHOI])

    print("\n=== Average Results ===")

    for name, datasets in results.items():
        for dataset, documents in datasets.items():
            print(f"\n{name} ({dataset}):")

            for metric, value in average(documents).items():
                print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
