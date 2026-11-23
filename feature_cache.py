"""缓存冻结主干的原图/水平翻转特征，仅加速分类层训练。"""

import hashlib
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from evaluate import ManifestDataset
from experiment_utils import CLASSES


class FlipFeatures(Dataset):
    def __init__(self, original, flipped, labels, augmentation):
        self.original, self.flipped, self.labels = original, flipped, labels
        self.augmentation = augmentation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        flip = self.augmentation == "horizontal_flip" and torch.rand(1).item() < 0.5
        return (self.flipped if flip else self.original)[index], self.labels[index]


def backbone_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith("fc."):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def encode_dataset(extractor, dataset, batch_size, device, both_views, logger):
    originals, flipped, labels = [], [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch, (images, targets) in enumerate(loader, 1):
        images = images.to(device)
        originals.append(extractor(images).cpu())
        if both_views:
            flipped.append(extractor(torch.flip(images, dims=[3])).cpu())
        labels.append(targets)
        if batch % 100 == 0:
            logger.info("Feature cache %s %d/%d", "train" if both_views else "val", batch, len(loader))
    return torch.cat(originals), torch.cat(flipped) if both_views else None, torch.cat(labels)


def feature_datasets(model, config, audit, cache_dir, device, logger):
    key = hashlib.sha256(("resnet18-224-imagenet-normalization-v1:" + backbone_digest(model)
        + audit["manifest_sha256"] + audit["dataset_sha256"]).encode()).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{key}.pt"
    # 构建缓存所需的 DataLoader 不得改变后续训练的随机顺序。
    rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
    try:
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            if cached["key"] != key:
                raise ValueError("Feature cache fingerprint mismatch")
            logger.info("Reusing frozen-backbone feature cache: %s", path)
        else:
            logger.info("Building frozen-backbone feature cache; train original/flip + validation, no test images")
            model.eval()
            extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten(1))
            original, flipped, labels = encode_dataset(extractor, ManifestDataset("train", CLASSES),
                config["batch_size"], device, True, logger)
            val_features, _, val_labels = encode_dataset(extractor, ManifestDataset("val", CLASSES),
                config["batch_size"], device, False, logger)
            cached = dict(key=key, original=original, flipped=flipped, labels=labels,
                          val_features=val_features, val_labels=val_labels)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as file:
                torch.save(cached, file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, path)
    finally:
        torch.set_rng_state(rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
    train = FlipFeatures(cached["original"], cached["flipped"], cached["labels"], config["augmentation"])
    val = TensorDataset(cached["val_features"], cached["val_labels"])
    return train, val, key
