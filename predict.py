"""Predict one local image with the same interface used by Streamlit."""
import argparse
import json

from PIL import Image

from garbage_classifier.common import project_path
from garbage_classifier.inference import Predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--checkpoint", default="checkpoints/B_frozen/best_resnet18.pth")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    args = parser.parse_args()
    predictor = Predictor(project_path(args.checkpoint), args.device)
    with Image.open(project_path(args.image)) as image:
        print(json.dumps(predictor.predict(image), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
