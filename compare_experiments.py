"""Regenerate comparison from completed experiments, without retraining."""
import argparse

from garbage_classifier.common import project_path
from garbage_classifier.comparison import compare


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="例如 results/A_scratch results/B_frozen results/C_finetune")
    parser.add_argument("--output", default="results/comparison")
    args = parser.parse_args()
    compare([project_path(path) for path in args.runs], project_path(args.output))
