"""Evaluate one checkpoint and export metrics, predictions and mistakes."""
import argparse

from garbage_classifier.common import project_path
from garbage_classifier.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/B_frozen/best_resnet18.pth")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", help="Default: results/<checkpoint experiment name>")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    checkpoint = project_path(args.checkpoint)
    output = project_path(args.output or f"results/{checkpoint.parent.name}")
    print(evaluate(checkpoint, project_path(args.data_dir), output, args.batch_size, args.device))


if __name__ == "__main__":
    main()
