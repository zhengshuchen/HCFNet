import torch
import torch.nn as nn

class Basicblock(nn.Module):
    "block for resnet18 and resnet34 the same with the original one"
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        out = self.relu(x)
        return out

class Bottleneck(nn.Module):
    "block for resnet 50 and more, switching the stride of conv1 and conv2 which is different with the original one"
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*4, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        out = self.relu(x)
        return out

class Resnet(nn.Module):
    def __init__(self, block, layers, basic_planes=64, dilations = [False, False, False]):
        super().__init__()
        "replace conv7x7 with 3 conv3x3 and replace self.in_planes from 64 to 128(did not do this) from Upernet Implementation"
        "change the basic_planes should change the channel of feat_maps but remember to keep the same with decoder in_planes"
        self.in_planes = 64
        self.dilation = 1
        self.basic_planes = basic_planes #the width of conv_layers , in 
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.conv2 = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.in_planes)
        self.conv3 = nn.Conv2d(self.in_planes, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.in_planes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.basic_planes, layers[0], stride=1)
        self.layer2 = self._make_layer(block, self.basic_planes*2, layers[1], stride=2, dilation=dilations[0])
        self.layer3 = self._make_layer(block, self.basic_planes*4, layers[2], stride=2, dilation=dilations[1])
        self.layer4 = self._make_layer(block, self.basic_planes*8, layers[3], stride=2, dilation=dilations[2])

    def _make_layer(self, block, planes, blocks, stride=1, dilation=False):
        downsample = None
        previous_dilation = self.dilation
        if dilation:
            self.dilation *= stride
            stride=1  
        if stride != 1 or planes * block.expansion != self.in_planes:
            'whether we should change the channel of residual original x'
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes*block.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*block.expansion)
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride=stride, downsample=downsample, dilation=previous_dilation))
        self.in_planes = planes * block.expansion
        for i in range(1,blocks):
            layers.append(block(self.in_planes, planes, stride=1, downsample=None, dilation=self.dilation))
        return nn.Sequential(*layers)
    def forward(self, x):
        feat_maps = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        feat_maps.append(x)
        x = self.layer2(x)
        feat_maps.append(x)
        x = self.layer3(x)
        feat_maps.append(x)
        x = self.layer4(x)
        feat_maps.append(x)
        return feat_maps

def resnet18_d(in_c=3, basic_planes=64):
    return Resnet(Basicblock, [2,2,2,2], basic_planes=basic_planes, dilations=[False, True, True])

def resnet34_d(in_c=3, basic_planes=64):
    return Resnet(Basicblock, [3,4,6,3], basic_planes=basic_planes, dilations=[False, True, True])

def resnet50_d(in_c=3, basic_planes=64):
    return Resnet(Bottleneck, [3,4,6,3], basic_planes=basic_planes, dilations=[False, True, True])

def resnet101_d(in_c=3, basic_planes=64):
    return Resnet(Bottleneck, [3,4,23,3], basic_planes=basic_planes, dilations=[False, True, True])

def resnet18(in_c=3, basic_planes=64):
    return  Resnet(Basicblock, [2,2,2,2], basic_planes=basic_planes, dilations=[False, False, False])

def resnet34(in_c=3, basic_planes=64):
    return  Resnet(Basicblock, [3,4,6,3], basic_planes=basic_planes, dilations=[False, False, False])

def resnet50(in_c=3, basic_planes=64):
    return Resnet(Bottleneck, [3,4,6,3], basic_planes=basic_planes, dilations=[False, False, False])

def resnet101(in_c=3, basic_planes=64):
    return Resnet(Bottleneck, [3,4,23,3], basic_planes=basic_planes, dilations=[False, False, False])

def main():
    net_18 = resnet18_d(3, 64)
    net_34 = resnet34_d(3, 64)
    net_50 = resnet50_d(3, 64)
    net_50_ = resnet50()
    x = torch.rand(2,3,512,512)
    y_18 = net_18(x)
    y_34 = net_34(x)
    y_50 = net_50(x)
    y_50_ = net_50_(x)
    # print(*y_18)
    for i in y_18:
        print(i.shape)
    # for i in y_34:
    #     print(i.shape)
    # for i in y_50:
    #     print(i.shape)
    # for i in y_50_:
    #     print(i.shape)
if __name__ == '__main__':
    main()