"""Run A/B/C sequentially on one fixed split, then create a comparison table."""
import argparse

from garbage_classifier.common import ROOT, project_path
from garbage_classifier.comparison import compare
from garbage_classifier.config import load_config
from garbage_classifier.engine import train
from garbage_classifier.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=["configs/A_scratch.json", "configs/B_frozen.json", "configs/C_finetune.json"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    args = parser.parse_args()
    configs = [load_config(path) for path in args.configs]
    if len({c.name for c in configs}) != len(configs):
        raise ValueError("实验名称不能重复")
    comparable = {(str(project_path(c.data_dir)), c.seed, c.image_size, c.batch_size, c.epochs, c.patience) for c in configs}
    if len(comparable) != 1:
        raise ValueError("对比实验需要相同的数据、种子、图像尺寸、batch_size、epochs 和 patience")
    # Preflight all destinations before starting a long run.
    for config in configs:
        for folder in ("checkpoints", "results"):
            path = ROOT / folder / config.name
            if path.exists() and any(path.iterdir()):
                raise FileExistsError(f"输出已存在：{path}，请更换实验名称")
    for config in configs:
        if args.device:
            config.device = args.device
        checkpoint = train(config)
        evaluate(checkpoint, project_path(config.data_dir), ROOT / "results" / config.name, config.batch_size, config.device)
    compare([ROOT / "results" / c.name for c in configs], ROOT / "results" / "comparison")


if __name__ == "__main__":
    main()
