"""ResNet-18 垃圾分类模型：定义结构和前向计算，不在这里训练。"""

import torch
from torch import nn
from torchvision.models import ResNet18_Weights
from torchvision.models.resnet import BasicBlock, ResNet


# ResNet-18 由四个阶段的残差模块组成，适合小型图像分类任务。
class GarbageClassifier(ResNet):
    """基于 ResNet-18 的分类网络，保留标准 ResNet 的权重名称。

    num_classes：类别数，TrashNet 预期为 6。
    pretrained：是否加载 ImageNet 预训练参数，首次启用可能下载权重。
    freeze_backbone：是否只训练最后的分类层；适用于已有预训练或本地权重时。
    """

    def __init__(self, num_classes=6, pretrained=False, freeze_backbone=False):
        if isinstance(num_classes, bool) or not isinstance(num_classes, int):
            raise TypeError("num_classes 必须是整数。")
        if num_classes < 2:
            raise ValueError("分类模型至少需要两个类别。")
        if not isinstance(pretrained, bool) or not isinstance(freeze_backbone, bool):
            raise TypeError("pretrained 和 freeze_backbone 必须是布尔值。")

        # BasicBlock 内部是两层卷积加残差连接；四个阶段各有两个模块。
        # 先保留 ImageNet 的 1000 类输出，才能完整加载其预训练参数。
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=1000)
        if pretrained:
            weights = ResNet18_Weights.DEFAULT.get_state_dict(
                progress=True, check_hash=True
            )
            self.load_state_dict(weights)

        # 主干提取 512 维特征，新的分类层输出 num_classes 个分数。
        self.fc = nn.Linear(self.fc.in_features, num_classes)
        self.freeze_features(freeze_backbone)

    def forward(self, images):
        """输入 [批次, 3, 高, 宽] 的浮点张量，输出 [批次, 类别数]。"""
        if not isinstance(images, torch.Tensor):
            raise TypeError("输入必须是经过预处理的 torch.Tensor。")
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("输入形状应为 [批次, 3, 高, 宽]，例如 [1, 3, 224, 224]。")
        if not images.is_floating_point():
            raise TypeError("输入必须是浮点张量，请先将图片转换为张量并标准化。")
        if images.shape[0] == 0 or min(images.shape[2:]) < 32:
            raise ValueError("批次不能为空，图片的高和宽至少为 32。")

        # 1. 初始卷积、归一化、激活和池化，提取基础纹理与边缘。
        # 下列尺寸注释以 224×224 输入为例。
        x = self.conv1(images)  # [N, 64, 112, 112]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # [N, 64, 56, 56]

        # 2. 四组残差模块逐步提取更复杂的图像特征。
        x = self.layer1(x)  # [N, 64, 56, 56]
        x = self.layer2(x)  # [N, 128, 28, 28]
        x = self.layer3(x)  # [N, 256, 14, 14]
        x = self.layer4(x)  # [N, 512, 7, 7]

        # 3. 全局平均池化，再把每张图片的特征展平成一维。
        x = self.avgpool(x)  # [N, 512, 1, 1]
        x = torch.flatten(x, 1)  # [N, 512]，保留第 0 维的批次。

        # 4. 返回未经 Softmax 的类别分数（logits）。
        # 后续训练的交叉熵直接接收 logits；app.py 预测时再转换为概率。
        return self.fc(x)

    def freeze_features(self, freeze=True):
        """freeze=True 只更新分类层；False 恢复整个网络的梯度。"""
        if not isinstance(freeze, bool):
            raise TypeError("freeze 必须是布尔值。")
        for name, parameter in self.named_parameters():
            parameter.requires_grad = not freeze or name.startswith("fc.")
        # 同步当前状态，避免冻结后仍更新 BatchNorm 的均值和方差。
        self.train(self.training)

    def train(self, mode=True):
        """保留 PyTorch 的训练/评估接口，同时固定已冻结的 BatchNorm。"""
        super().train(mode)
        if mode:
            for module in self.modules():
                # 训练模式下，冻结的 BatchNorm 不更新均值和方差。
                if isinstance(module, nn.BatchNorm2d) and not any(
                    parameter.requires_grad for parameter in module.parameters()
                ):
                    module.eval()
        return self


def build_model(num_classes=6, pretrained=False, freeze_backbone=False):
    """统一创建模型；默认随机初始化，页面随后加载本地训练权重。"""
    return GarbageClassifier(num_classes, pretrained, freeze_backbone)
