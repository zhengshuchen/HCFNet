import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Position_attention(nn.Module):
    def __init__(self, in_c, mid_c=None):
        super().__init__()
        mid_c = mid_c or in_c // 8
        self.q = nn.Conv2d(in_c, mid_c, kernel_size=1)
        self.k = nn.Conv2d(in_c, mid_c, kernel_size=1)
        self.v = nn.Conv2d(in_c, in_c, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, _, h, w = x.shape
        q = self.q(x).view(b, -1, h * w).permute(0, 2, 1)  # bs, hw, c
        k = self.k(x).view(b, -1, h * w)  # bs, c ,hw
        v = self.v(x).view(b, -1, h * w)  # bs, c, hw
        # att = self.softmax(q @ k)
        att = self.softmax(torch.bmm(q,k))
        # out = (v @ att.permute(0, 2, 1)).view(b, -1, h, w)
        out = torch.bmm(v, att.permute(0, 2, 1)).view(b, -1, h, w)
        out = self.gamma * out + x

        return out


class Channel_attention(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.in_c = in_c
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, _, h, w = x.shape
        q = x.view(b, -1, h * w)  # bs, c ,hw
        k = x.view(b, -1, h * w).permute(0, 2, 1)  # bs, hw, c
        v = x.view(b, -1, h * w)  # bs, c, hw
        att = self.softmax(q @ k)  # b, c, c
        out = att @ v
        out = out.view(b, -1, h, w)
        out = self.gamma * out + x
        return out


class Double_attention(nn.Module):
    def __init__(self, in_c, mid_c=None):
        super().__init__()
        self.pam = Position_attention(in_c, mid_c)
        self.cam = Channel_attention(in_c)
        self.relu = nn.ReLU()

    def forward(self, x):
        pam_out = self.pam(x)
        cam_out = self.cam(x)
        return pam_out + cam_out


class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''

    def __init__(self, c):
        super(External_attention, self).__init__()

        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h * w
        x = x.view(b, c, h * w)  # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x)
        return x

def main():
    x = torch.rand(3,512,64,64)
    EA = External_attention(256)
    out = EA(x)
    print(out.shape)

if __name__ == '__main__':
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    model = Double_attention(512)
    x = torch.rand(1,512,32,32)
    flopts = FlopCountAnalysis(model, x)
    print('FLOPS: ',flopts.total())
    print('PARAMS: ', parameter_count_table(model))
    import ptflops
    GMacs,Params = ptflops.get_model_complexity_info(model, (512,32,32))
    print(GMacs, Params)