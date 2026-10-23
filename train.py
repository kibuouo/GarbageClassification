"""Run one configured ResNet-18 experiment."""
import argparse

from garbage_classifier.config import load_config
from garbage_classifier.engine import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/B_frozen.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--name", help="Override experiment name to retain existing runs")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.device = args.device
    if args.name:
        config.name = args.name
    print(f"最佳模型：{train(config)}")


if __name__ == "__main__":
    main()
