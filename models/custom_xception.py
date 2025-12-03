import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SeparableConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()

        self.residual = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.bn_res = nn.BatchNorm2d(out_channels)

        self.relu1 = nn.ReLU(inplace=False)
        self.sep1 = SeparableConv2d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu2 = nn.ReLU(inplace=False)
        self.sep2 = SeparableConv2d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.pool = nn.MaxPool2d(3, stride=stride, padding=1)

    def forward(self, x):
        res = self.bn_res(self.residual(x))

        out = self.relu1(x)
        out = self.bn1(self.sep1(out))
        out = self.relu2(out)
        out = self.bn2(self.sep2(out))

        out = self.pool(out)
        return out + res

class MiniXception(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.entry = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False)
        )

        self.block1 = SeparableConvBlock(64, 128)
        self.block2 = SeparableConvBlock(128, 256)
        self.block3 = SeparableConvBlock(256, 512)

        self.exit_sep = SeparableConv2d(512, 1024)
        self.exit_bn = nn.BatchNorm2d(1024)
        self.exit_relu = nn.ReLU(inplace=False)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.exit_relu(self.exit_bn(self.exit_sep(x)))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)