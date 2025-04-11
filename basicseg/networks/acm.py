import torch
import torch.nn as nn
import torch.nn.functional as F
from basicseg.utils.registry import NET_REGISTRY

class BiLocalChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(BiLocalChaFuseReduce, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid()
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)
        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)

        out = 2 * xl * topdown_wei + 2* xh * bottomup_wei
        out = self.post(out)
        return out


class AsymBiChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(AsymBiChaFuseReduce, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * xl * topdown_wei + 2 * xh * bottomup_wei
        out = self.post(xs)
        return out


class BiGlobalChaFuseReduce(nn.Module):
    def __init__(self, in_high_channels, in_low_channels, out_channels=64, r=4):
        super(BiGlobalChaFuseReduce, self).__init__()

        assert in_low_channels == out_channels
        self.high_channels = in_high_channels
        self.low_channels = in_low_channels
        self.out_channels = out_channels
        self.bottleneck_channels = int(out_channels // r)

        self.feature_high = nn.Sequential(
            nn.Conv2d(self.high_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.topdown = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.out_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.bottomup = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.low_channels, self.bottleneck_channels, 1, 1, 0),
            nn.BatchNorm2d(self.bottleneck_channels),
            nn.ReLU(True),

            nn.Conv2d(self.bottleneck_channels, self.out_channels, 1, 1, 0),
            nn.BatchNorm2d(self.out_channels),
            nn.Sigmoid(),
        )

        self.post = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(True),
        )

    def forward(self, xh, xl):
        xh = self.feature_high(xh)

        topdown_wei = self.topdown(xh)
        bottomup_wei = self.bottomup(xl)
        xs = 2 * xl * topdown_wei + 2 * xh * bottomup_wei
        out = self.post(xs)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = F.relu(x+residual, True)
        return out


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)

@NET_REGISTRY.register()
class ASKCResNetFPN(nn.Module):
    def __init__(self, layer_blocks=[4,4,4], channels=[8,16,32,64], fuse_mode='AsymBi'):
        super(ASKCResNetFPN, self).__init__()

        stem_width = channels[0]
        self.stem = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, stem_width*2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width*2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[1], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        self.fuse23 = self._fuse_layer(channels[3], channels[2], channels[2], fuse_mode)
        self.fuse12 = self._fuse_layer(channels[2], channels[1], channels[1], fuse_mode)

        self.head = _FCNHead(channels[1], 1)

    def forward(self, x):
        _, _, hei, wid = x.shape

        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        out = self.layer3(c2)

        out = F.interpolate(out, size=[hei//8, wid//8], mode='bilinear')
        out = self.fuse23(out, c2)

        out = F.interpolate(out, size=[hei//4, wid//4], mode='bilinear')
        out = self.fuse12(out, c1)

        pred = self.head(out)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear')

        return out

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        downsample = (in_channels != out_channels) or (stride != 1)
        layer = []
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):
        assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        if fuse_mode == 'BiLocal':
            fuse_layer = BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'BiGlobal':
            fuse_layer = BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer

@NET_REGISTRY.register()
class ASKCResUNet(nn.Module):
    def __init__(self, layer_blocks=[4,4,4], channels=[8,16,32,64], fuse_mode='AsymBi'):
        super(ASKCResUNet, self).__init__()

        stem_width = int(channels[0])
        self.stem = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, stem_width, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True),

            nn.Conv2d(stem_width, 2*stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(2*stem_width),
            nn.ReLU(True),

            nn.MaxPool2d(3, 2, 1),
        )

        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[1], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        self.deconv2 = nn.ConvTranspose2d(channels[3], channels[2], 4, 2, 1)
        self.fuse2 = self._fuse_layer(channels[2], channels[2], channels[2], fuse_mode)
        self.uplayer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                         in_channels=channels[2], out_channels=channels[2], stride=1)

        self.deconv1 = nn.ConvTranspose2d(channels[2], channels[1], 4, 2, 1)
        self.fuse1 = self._fuse_layer(channels[1], channels[1], channels[1], fuse_mode)
        self.uplayer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                         in_channels=channels[1], out_channels=channels[1], stride=1)

        self.head = _FCNHead(channels[1], 1)

    def forward(self, x):
        _, _, hei, wid = x.shape

        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        deconc2 = self.deconv2(c3)
        fusec2 = self.fuse2(deconc2, c2)
        upc2 = self.uplayer2(fusec2)

        deconc1 = self.deconv1(upc2)
        fusec1 = self.fuse1(deconc1, c1)
        upc1 = self.uplayer1(fusec1)

        pred = self.head(upc1)
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear')
        return out

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layer = []
        downsample = (in_channels != out_channels) or (stride != 1)
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)

    def _fuse_layer(self, in_high_channels, in_low_channels, out_channels, fuse_mode='AsymBi'):
        assert fuse_mode in ['BiLocal', 'AsymBi', 'BiGlobal']
        if fuse_mode == 'BiLocal':
            fuse_layer = BiLocalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'AsymBi':
            fuse_layer = AsymBiChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        elif fuse_mode == 'BiGlobal':
            fuse_layer = BiGlobalChaFuseReduce(in_high_channels, in_low_channels, out_channels)
        else:
            NameError
        return fuse_layer

def main():
    import time
    from thop import profile
    from thop import clever_format  # 用于格式化输出的 MACs 和参数数量
    layer_blocks = [4,4,4]
    channels = [8, 16, 32, 64]
    net = ASKCResUNet(layer_blocks,  channels, fuse_mode='AsymBi')
    # 定义输入张量
    input_tensor = torch.randn(1, 3, 1024, 1024)  # 假设输入为 (batch_size=1, 3通道, 512x512 图像)

    # 检查当前设备并将模型移动到相应设备
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    input_tensor = input_tensor.to(device)

    # 使用 thop 计算 MACs 和参数量
    flops, params = profile(net, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.2f")

    # 打印计算成本和参数量
    print(f"Computational cost (MACs): {flops}")
    print(f"Number of parameters: {params}")

    # # # 测试 100 张图片的推理时间
    # total_time = 0
    # num_images = 100
    #
    # # 确保模型处于评估模式
    # net.eval()
    #
    # with torch.no_grad():  # 禁用梯度计算以提高推理速度
    #     for _ in range(num_images):
    #         torch.cuda.synchronize()  # 同步 GPU 和 CPU，确保时间精确
    #         start = time.time()
    #         result = net(input_tensor)
    #         torch.cuda.synchronize()
    #         end = time.time()
    #
    #         infer_time = end - start
    #         total_time += infer_time
    #
    #         # print(f'Single inference time: {infer_time:.6f} seconds')
    #
    # # 计算平均推理时间和 FPS
    # average_time = total_time / num_images
    # fps = 1 / average_time if average_time > 0 else float('inf')
    #
    # print(f'Average inference time for 100 images: {average_time:.6f} seconds')
    # print(f'FPS: {fps:.2f}')
if __name__ == '__main__':
    main()