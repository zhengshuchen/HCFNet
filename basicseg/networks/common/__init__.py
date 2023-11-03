from basicseg.networks.common.resnet import resnet18, resnet18_d, resnet34, resnet34_d,\
    resnet50, resnet50_d, resnet101, resnet101_d
from basicseg.networks.common.attention import Double_attention, \
    Position_attention, Channel_attention
from basicseg.networks.common.layernorm import LayerNorm2d
from basicseg.networks.common.conv import CDC_conv, ASPP, GatedConv2dWithActivation, DeformConv2d
# from torch
import torch.nn as nn

def main():
    pass

if __name__ == '__main__':
    main()
# def convert_conv2d(model, in_module, out_module, **kwargs):
#     model_output = model
#     if isinstance(model, in_module):
#         model_output = out