import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    实现子module: Residual Block
    """

    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.se = SELayer(outchannel, 16)
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        out = self.se(out)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet(nn.Module):
    """
    实现主module：ResNet34
    ResNet34包含多个layer，每个layer又包含多个Residual block
    用子module来实现Residual block，用_make_layer函数来实现layer
    """

    def __init__(self, blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.model_name = 'resnet34'

        # 前几层: 图像转换
        self.pre = nn.Sequential(
            nn.Conv2d(4, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))

        # 重复的layer，分别有3，4，6，3个residual block
        self.layer1 = self._make_layer(64, 64, blocks[0])
        self.layer2 = self._make_layer(64, 128, blocks[1], stride=2)
        self.layer3 = self._make_layer(128, 256, blocks[2], stride=2)
        self.layer4 = self._make_layer(256, 512, blocks[3], stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建layer,包含多个residual block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU()
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return x


def Se_ResNet18():
    return ResNet([2, 2, 2, 2])




# 模型
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.ResNet18_model = ResNet([2, 2, 2, 2])
        self.Line1 = nn.Linear(512, 256)
        self.Line2 = nn.Linear(256, 1)

    def forward_once(self, input1, input2):
        output1 = self.ResNet18_model(input1)
        output2 = self.ResNet18_model(input2)
        output3 = output1-output2
        return output3

    def forward(self, input1, input2):
        output1= self.forward_once(input1, input2)
        output2 = self.Line1(output1)
        output = self.Line2(output2)
        output = output.squeeze(-1)
        return output


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):

        loss_contrastive = F.mse_loss(input1, input2)

        return loss_contrastive