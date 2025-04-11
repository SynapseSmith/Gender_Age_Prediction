import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        reduced_channels = in_channels // reduction
        self.fc1 = nn.Conv2d(in_channels, reduced_channels, 1)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, 1)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_factor

        self.expand_conv = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(mid_channels)
        self.depthwise_conv = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, groups=mid_channels,
                                        bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.se = SqueezeExcitation(mid_channels)
        self.project_conv = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.swish = Swish()

        self.skip_connection = (in_channels == out_channels) and (stride == 1)

    def forward(self, x):
        residual = x

        x = self.swish(self.bn0(self.expand_conv(x)))
        x = self.swish(self.bn1(self.depthwise_conv(x)))
        x = self.se(x)
        x = self.bn2(self.project_conv(x))

        if self.skip_connection:
            x += residual

        return x


class EfficientNetB0(nn.Module):
    def __init__(self):
        super(EfficientNetB0, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 1, 1),
            MBConvBlock(16, 24, 6, 2),
            MBConvBlock(24, 24, 6, 1),
            MBConvBlock(24, 40, 6, 2),
            MBConvBlock(40, 40, 6, 1),
            MBConvBlock(40, 80, 6, 2),
            MBConvBlock(80, 80, 6, 1),
            MBConvBlock(80, 80, 6, 1),
            MBConvBlock(80, 112, 6, 1),
            MBConvBlock(112, 112, 6, 1),
            MBConvBlock(112, 112, 6, 1),
            MBConvBlock(112, 192, 6, 2),
            MBConvBlock(192, 192, 6, 1),
            MBConvBlock(192, 192, 6, 1),
            MBConvBlock(192, 192, 6, 1),
            MBConvBlock(192, 320, 6, 1)
        )

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            Swish()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

class GenderAgeModel(nn.Module):
    def __init__(self):
        super(GenderAgeModel, self).__init__()
        self.efficientnet = EfficientNetB0()
        self.fc_gender = nn.Linear(1280, 2)
        self.fc_age = nn.Linear(1280, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.efficientnet(x)
        gender_out = self.fc_gender(x)
        age_out = self.fc_age(x)
        return gender_out, age_out
