# 迁移学习是通过把预训练参数迁移给我们自己搭建的相同命名的参数，所以要注意参数的命名和源码中命名保持一致
import torch
from model import BrainEmotionConvLayer

# resnet网络分成四个大的卷积模块，每个卷积模块中都包含了好几个残差结构，下一个卷积模块的长宽尺寸都是上一个卷积模块尺寸的一半
# 每个卷积模块中的残差结构都是通过实线跳连接的，而每个卷积模块之间都是通过虚线进行跳连接的
# 这个虚线的跳连接就是下面设置的downsample参数，也就是下采样操作，将长宽尺寸缩小一半


class BasicBlock(torch.nn.Module):  # 18/34层网络的残差结构
    expansion = 1  # 18/34层网络的每个残差结构中的两个卷积层卷积核个数都是一样的，所以等于1
    # 但是对于50/101/152层网络的每个残差结构，第三个卷积层的卷积核个数是前两个卷积层的卷积核个数的4倍，所以对于50/101/152层网络expansion应该等于4

    # downsample为下采样操作（即虚线连接的结构），可以看到残差网络的每个卷积模块到下一个卷积模块之间都进行了一步下采样操作，使图像的尺寸变成原来的一半
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, AddBrain=False):
        super(BasicBlock, self).__init__()
        
        self.AddBrain = AddBrain
        # 不使用偏置参数，对于BN来说，使不使用偏置参数都一样
        if self.AddBrain:
            self.conv1 = BrainEmotionConvLayer(input_dim=in_channels, out_dim=out_channels, embed_dim=out_channels)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.conv2 = BrainEmotionConvLayer(input_dim=out_channels, out_dim=out_channels, embed_dim=out_channels)
            self.relu = torch.nn.ReLU(inplace=True)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)  # BN操作一般放在卷积和激活函数之间
            # 即虚线结构（对34层结构，是一个1x1的卷积核，步长是2，这个下采样分支的channel和非跳连接支路的输出通道保持相等）
            self.downsample = downsample
        else:
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False) 
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
            self.relu = torch.nn.ReLU(inplace=True)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)  # BN操作一般放在卷积和激活函数之间
            # 即虚线结构（对34层结构，是一个1x1的卷积核，步长是2，这个下采样分支的channel和非跳连接支路的输出通道保持相等）
            self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x = self.downsample(x)  # 如果downsample不是None，则进行下采样操作

        out += x
        out = self.relu(out)

        return out


class Bottleneck(torch.nn.Module):  # 50/101/152层网络的残差结构
    expansion = 4  # 对于50/101/152层网络的每个残差结构，第三个卷积层的卷积核个数是前两个卷积层的卷积核个数的4倍，所以对于50/101/152层网络expansion应该等于4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = torch.nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = torch.nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = torch.nn.Conv2d(in_channels=width, out_channels=width, groups=groups, kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = torch.nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion, kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = torch.nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = torch.nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(torch.nn.Module):
    # block参数代表是选择使用18/34层网络的残差结构，还是使用50/101/152层网络的残差结构
    # blocks_num是一个列表，每个数表示该卷积模块包含的残差结构数量，例如：如果是34层网络，那么就是[3, 4, 6, 3];
    # 如果是101层网络，那就是[3, 4, 23, 3]
    # include_top参数是为了方便在resnet网络基础上搭建更复杂的网络结构
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True, AddBrain=False):
        super(ResNet, self).__init__()
        
        self.AddBrain = AddBrain
        self.include_top = include_top
        self.in_channels = 64

        self.conv1 = torch.nn.Conv2d(3, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 构建四个卷积模块的第一个残差结构
        # 下面的64，128,256,512代表的是每个卷积模块的第一个残差结构的输入
        # 第一个卷积模块的第一个残差结构输入的stride就等于1
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # 从第二个卷积模块开始，第一个残差结构输入的stride等于2，因为从第二个卷积模块开始，每个卷积模块的第一个残差结构都是虚线跳连接，有一个下采样的操作
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))  # 不管图像的长宽尺寸是多少，经过平均池化后都等于长宽1x1的
            self.fc = torch.nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, channel, block_num, stride=1):  # channel是一个卷积模块中的一个残差结构的输入通道数
        downsample = None  # stride等于1的时候，是实线跳连接，不进行下采样操作
        # 18/34层网络结构的第一个卷积模块的第一个残差结构是实线跳连接，但是50/101/152层网络结构的第一个卷积模块的第一个残差结构是虚线跳连接，但是不改变图像高和宽的尺寸，只改变深度
        # 当stride等于2，也就是虚线跳连接，或者是50/101/152层网络的第一个卷积模块的第一个残差结构时（因为只有50/101/152层网络block.expansion=4，self.in_channels != channel * block.expansion）采用下采样操作
        if stride != 1 or self.in_channels != channel * block.expansion:
            # 下采样操作实际就是一个1x1大小、步长为2的卷积操作+一个BN操作（50/101/152层网络的第一个卷积模块的第一个残差结构步长还是1，只改变通道深度，不改变图像尺寸）
            downsample = torch.nn.Sequential(
                torch.nn.Conv2d(self.in_channels, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channels, channel, downsample=downsample, stride=stride, AddBrain=False))  # 先把第一个残差结构加到layers
        # 50/101/152层网络经过第一个残差结构之后，通道深度会变成channel * 4，所以加一步这个操作
        self.in_channels = channel * block.expansion

        # 因为第一个残差结构已经加入到layers了，所以这里是从1开始，而不是从0开始，这里是把后面的实线残差结构也加入layers
        for _ in range(1, block_num):
            # 因为后面几个残差结构都是实线，所以就不需要下采样操作了，stride也就等于1
            layers.append(block(self.in_channels, channel, AddBrain=self.AddBrain))

        # 之所以layers前面加*是表示采用非关键字参数传入，具体可以看OneNote笔记torch.nn.Sequential()部分对两种方法的解释
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet34(num_classes=1000, include_top=True, AddBrain=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top, AddBrain=AddBrain)
 
 
def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
