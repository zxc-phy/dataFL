import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(nn.ReLU(inplace=True)(x))
        out = self.conv2(nn.ReLU(inplace=True)(out))
        out = torch.cat([out, x], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        # self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(nn.ReLU(inplace=True)(x))
        out = self.pool(out)
        return out

class DenseNet(nn.Module):
    def __init__(self, num_blocks, growth_rate=32, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm2d(num_channels)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)

        self.dense1 = self._make_dense_layers(num_channels, num_blocks[0], growth_rate)
        num_channels += num_blocks[0] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans1 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense2 = self._make_dense_layers(num_channels, num_blocks[1], growth_rate)
        num_channels += num_blocks[1] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans2 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense3 = self._make_dense_layers(num_channels, num_blocks[2], growth_rate)
        num_channels += num_blocks[2] * growth_rate
        out_channels = int(num_channels * reduction)
        self.trans3 = Transition(num_channels, out_channels)
        num_channels = out_channels

        self.dense4 = self._make_dense_layers(num_channels, num_blocks[3], growth_rate)
        num_channels += num_blocks[3] * growth_rate

        # self.bn2 = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

    def _make_dense_layers(self, in_channels, n_blocks, growth_rate):
        layers = []
        for _ in range(n_blocks):
            layers.append(Bottleneck(in_channels, growth_rate))
            in_channels += growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pool1(nn.ReLU(inplace=True)(self.conv1(x)))
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = nn.ReLU(inplace=True)(out)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def densenet(num_classes=10):
    return DenseNet(num_blocks=[6, 12, 24, 16], growth_rate=32, num_classes=num_classes)