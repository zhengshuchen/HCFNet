import torch
import torch.nn as nn

import time
from thop import profile
from thop import clever_format  # 用于格式化输出的 MACs 和参数数量
import torch.nn.functional as F
# from ptflops import get_model_complexity_info
from basicseg.main_blocks import PPA, DASI, MDCR
from basicseg.utils.registry import NET_REGISTRY


@NET_REGISTRY.register()
class HCFnet(nn.Module):
    def __init__(self,
                in_features=3,
                out_features=1,
                gt_ds = False,
                ) -> None:
        super().__init__()
        self.gt_ds = gt_ds

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.p1 = PPA(in_features=in_features,
                                filters=32)

        self.respath1 = DASI(in_features=32,
                                out_features=32,
                               )
        self.p2 = PPA(in_features=32,
                                filters=int(32 * 2))
        self.respath2 = DASI(in_features=64,
                                out_features=32 * 2,
                                )
        self.p3 = PPA(in_features=64,
                                filters=int(32 * 4))
        self.respath3 = DASI(in_features=128,
                                out_features=32 * 4,
                                )
        self.p4 = PPA(in_features=128,
                                filters=int(32 * 8))
        self.respath4 = DASI(in_features=256,
                                out_features=32 * 8,
                                )
        self.p5 = PPA(in_features=256,
                                filters=int(32 * 16))

        self.mdcr = MDCR(in_features=int(512),out_features=int(512))



        self.up1 = nn.Sequential(nn.ConvTranspose2d(512,
                                      32*8,
                                    kernel_size=(2,2),
                                    stride=(2,2)),
                                    nn.BatchNorm2d(32 * 8),
                                    nn.ReLU())


        self.p6 = PPA(in_features=32 * 8 * 2,
                                filters=int(32 * 8))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(256,
                                      32*4,
                                    kernel_size=(2,2),
                                    stride=(2,2)),
                                    nn.BatchNorm2d(32 * 4),
                                    nn.ReLU())

        self.p7 = PPA(in_features=32 * 4 * 2,
                                filters=int(32 * 4))

        self.up3 = nn.Sequential(nn.ConvTranspose2d(128,
                                      32*2,
                                    kernel_size=(2,2),
                                    stride=(2,2)),
                                    nn.BatchNorm2d(32 * 2),
                                    nn.ReLU())

        self.p8 = PPA(in_features=32 * 2 * 2,
                                filters=int(32 * 2))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(64,
                                      32,
                                    kernel_size=(2,2),
                                    stride=(2,2)),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU())

        self.p9 = PPA(in_features=32 * 2,
                                filters=int(32))


        self.out = nn.Conv2d(in_channels=32,
                             out_channels=out_features,
                             kernel_size=(1, 1),
                             padding=(0, 0)
                            )
        self.out1 =  nn.Conv2d(in_channels=512,
                      out_channels=out_features,
                      kernel_size=(1, 1),
                      padding=(0, 0)
                      )
        self.out2 = nn.Conv2d(in_channels=256,
                              out_channels=out_features,
                              kernel_size=(1, 1),
                              padding=(0, 0)
                              )
        self.out3 = nn.Conv2d(in_channels=128,
                              out_channels=out_features,
                              kernel_size=(1, 1),
                              padding=(0, 0)
                              )
        self.out4 = nn.Conv2d(in_channels=64,
                              out_channels=out_features,
                              kernel_size=(1, 1),
                              padding=(0, 0)
                              )



    def forward(self, x):
        #encoder
        x1 = self.p1(x)
        xp1 = self.maxpool(x1)
        x2 = self.p2(xp1)
        xp2 = self.maxpool(x2)
        x3 = self.p3(xp2)
        xp3 = self.maxpool(x3)
        x4 = self.p4(xp3)
        xp4 = self.maxpool(x4)
        x = self.p5(xp4)
        x = self.mdcr(x)


        x1_res = self.respath1(x1, x2, None) # 1 32 512 512
        x2_res = self.respath2(x2, x3, x1) # 1 64 256 256
        x3_res = self.respath3(x3, x4, x2) # 1 128 128 128
        x4_res = self.respath4(x4,  x, x3) # 1 256 64 64


        #decoder
        out4 = F.interpolate(self.out1(x), scale_factor=16, mode ='bilinear', align_corners=True)
        x = self.up1(x)
        x = torch.cat((x, x4_res), dim=1)
        x = self.p6(x)
        out3 = F.interpolate(self.out2(x), scale_factor=8, mode ='bilinear', align_corners=True)
        x = self.up2(x)
        x = torch.cat((x, x3_res), dim=1)
        x = self.p7(x)
        out2 = F.interpolate(self.out3(x), scale_factor=4, mode ='bilinear', align_corners=True)
        x = self.up3(x)
        x = torch.cat((x, x2_res), dim=1)
        x = self.p8(x)
        out1 = F.interpolate(self.out4(x), scale_factor=2, mode ='bilinear', align_corners=True)
        x = self.up4(x)
        x = torch.cat((x, x1_res), dim=1)
        x = self.p9(x)
        out = self.out(x)
        if self.gt_ds:
            return out,out1,out2,out3,out4
        else:
            return out



if __name__ == '__main__':
    # 定义输入张量
    input_tensor = torch.randn(1, 3, 1024, 1024)  # 假设输入为 (batch_size=1, 3通道, 512x512 图像)

    # 实例化模型 (确保你的 `SAMamba` 模型类定义正确)
    net = HCFnet()
    # print(net.encoder)

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

    # # 测试 100 张图片的推理时间
    # total_time = 0
    # num_images = 100

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

