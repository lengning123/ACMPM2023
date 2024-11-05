import torch
import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 卷积用于压缩通道数
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3x3 卷积用于提取空间特征
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1x1 卷积用于恢复通道数
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)

        self.relu = nn.ReLU(inplace=True)

        # 跳跃连接可能需要降采样
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # 如果有 downsample，需要对 identity 做投影
        if self.downsample is not None:
            x = self.downsample(identity)

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)


        out += x
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()

        # 初始的 7x7 卷积 + 最大池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差模块组
        self.layer1 = self._make_layer(64, 64, 3, stride=1)  # 第一组：64 通道，3 个残差块
        self.layer2 = self._make_layer(64, 128, 4, stride=2)  # 第二组：128 通道，4 个残差块
        self.layer3 = self._make_layer(128, 256, 6, stride=2)  # 第三组：256 通道，6 个残差块
        self.layer4 = self._make_layer(256, 512, 3, stride=2)  # 第四组：512 通道，3 个残差块

        # 全局平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 1, num_classes)  # 512 * 4 是因为最后的输出通道数为 512 * 4

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        downsample = None
        # 第一个残差块的降采样，使用 1x1 卷积来调整通道数和尺寸
        if stride != 1 or in_channels != out_channels/2:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        # 第一个块，可能需要降采样
        layers.append(Bottleneck(out_channels, out_channels//4, stride, downsample))

        # 后续的块，不需要降采样，直接连接
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, out_channels//4))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

###test

'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 实例化模型并迁移到 GPU
model = ResNet50(num_classes=10).to(device)

# 测试模型的输出，假设输入大小为 224x224
x = torch.randn(1, 3, 224, 224).to(device)
output = model(x)
print(output)  # 应该输出 torch.Size([1, 1000])
'''
