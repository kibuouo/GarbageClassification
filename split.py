"""将垃圾图片按类别随机划分，保留 raw 中的原始文件。"""

import csv
import random
import shutil
from pathlib import Path

# 以脚本所在位置定位项目目录。
DATA = Path(__file__).resolve().parent / "data"
RAW = DATA / "raw" / "dataset-resized"

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# 这些图片内容重复且标签冲突，暂不用于训练或评估。
# 只跳过复制，不删除 raw 中的原始图片。
EXCLUDED = {
    "glass/glass115.jpg",
    "metal/metal91.jpg",
    "glass/glass176.jpg",
    "plastic/plastic152.jpg",
    "glass/glass389.jpg",
    "plastic/plastic332.jpg",
}

# 提前检查目标目录，防止重复运行时混入旧的划分。
for split in ["train", "val", "test"]:
    folder = DATA / split
    if folder.exists() and any(p.is_file() for p in folder.rglob("*")):
        raise FileExistsError(f"{folder} 已有文件，请检查后再运行。")

# 固定随机种子，保证相同图片列表得到相同划分。
rng = random.Random(42)
records = []

for label in CLASSES:
    # 先排序再打乱，避免文件系统返回顺序影响划分结果。
    images = sorted(
        p
        for p in (RAW / label).glob("*.jpg")
        if p.relative_to(RAW).as_posix() not in EXCLUDED
    )
    rng.shuffle(images)

    # 验证集、测试集各取约 15%，剩余图片进入训练集。
    n_val = round(len(images) * 0.15)
    n_test = round(len(images) * 0.15)
    n_train = len(images) - n_val - n_test

    groups = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    for split, files in groups.items():
        destination = DATA / split / label
        destination.mkdir(parents=True, exist_ok=True)

        for image in files:
            shutil.copy2(image, destination / image.name)
            records.append([split, label, image.name])

    print(f"{label}: 训练 {n_train}，验证 {n_val}，测试 {n_test}")

# 保存划分清单，之后可以查看每张图片分到了哪里。
with (DATA / "split.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file, lineterminator="\n")
    writer.writerow(["split", "class", "filename"])
    writer.writerows(records)

print("划分完成，清单已保存到 data/split.csv")
