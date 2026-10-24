"""训练垃圾分类模型，使用方式：python train.py。"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_model

# 路径统一以当前脚本所在目录为基准。
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CHECKPOINT = ROOT / "checkpoints" / "best_resnet18.pth"

# 先用少量轮次跑通流程，后续再调整。
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 0.001


def run_epoch(model, loader, criterion, device, optimizer=None):
    """执行一轮训练或验证。

    传入 optimizer：执行训练，更新参数。
    不传 optimizer：执行验证，只计算损失和准确率。
    """
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in loader:
        # 图片和标签必须放在与模型相同的设备上。
        images = images.to(device)
        labels = labels.to(device)

        if optimizer is not None:
            # 清空上一批图片产生的梯度，避免重复累加。
            optimizer.zero_grad(set_to_none=True)

        # 训练时计算梯度；验证时关闭梯度计算。
        with torch.set_grad_enabled(training):
            outputs = model(images)

            # 模型直接输出类别分数，这里不要先做 Softmax。
            loss = criterion(outputs, labels)

            if optimizer is not None:
                loss.backward()  # 计算参数应该如何调整。
                optimizer.step()  # 根据梯度更新可训练参数。

        # 按图片数量累计，最后一批不足 batch size 时也能正确统计。
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        # 分数最大的类别编号，就是当前预测。
        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_images += batch_size

    average_loss = total_loss / total_images
    accuracy = total_correct / total_images
    return average_loss, accuracy


def main():
    # 固定随机种子，让后续重复运行的随机过程尽量一致。
    torch.manual_seed(42)

    # NVIDIA 显卡使用 CUDA，Apple Silicon 使用 MPS，否则使用 CPU。
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"训练设备：{device}", flush=True)

    # 训练集增加随机水平翻转，让模型接触不同方向的图片。
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )

    # 验证集不做随机增强，处理规则与 app.py 的预测保持一致。
    val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )

    # ImageFolder 根据文件夹名称自动生成类别编号。
    train_data = datasets.ImageFolder(
        DATA / "train",
        transform=train_transform,
    )
    val_data = datasets.ImageFolder(
        DATA / "val",
        transform=val_transform,
    )

    if train_data.class_to_idx != val_data.class_to_idx:
        raise ValueError("训练集和验证集的类别文件夹不一致。")

    # 训练时打乱图片顺序；验证时保持固定顺序。
    # num_workers=0 先使用最简单的数据加载方式。
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"训练图片：{len(train_data)} 张")
    print(f"验证图片：{len(val_data)} 张")
    print(f"类别顺序：{train_data.classes}", flush=True)

    # 使用 ImageNet 预训练参数，只训练最后的分类层。
    # 首次运行可能需要下载预训练权重。
    model = build_model(
        num_classes=len(train_data.classes),
        pretrained=True,
        freeze_backbone=True,
    ).to(device)

    # 多类别分类使用交叉熵损失。
    criterion = nn.CrossEntropyLoss()

    # 只把允许更新的参数交给优化器。
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(1, EPOCHS + 1):
        # 传入优化器，执行一轮真正的训练。
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        # 不传优化器，只进行验证，不更新参数。
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)

        print(
            f"第 {epoch}/{EPOCHS} 轮 | "
            f"训练 loss={train_loss:.4f}，准确率={train_accuracy:.2%} | "
            f"验证 loss={val_loss:.4f}，准确率={val_accuracy:.2%}",
            flush=True,
        )

        # 只在验证准确率提升时保存，避免最后一轮覆盖更好的模型。
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            # 转为 CPU 张量保存，方便之后在不同设备上加载。
            state = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }

            # 字段名称与 app.py 中的加载方式保持一致。
            torch.save(
                {
                    "model_state": state,
                    "classes": train_data.classes,
                },
                CHECKPOINT,
            )
            print(f"已保存最佳模型：{CHECKPOINT}", flush=True)

    print(f"训练结束，最佳验证准确率：{best_accuracy:.2%}")


# 直接运行 train.py 才开始训练，导入这个文件时不会自动训练。
if __name__ == "__main__":
    main()
