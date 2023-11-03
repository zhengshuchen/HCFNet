import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['NonLocalBlock', 'GCA_Channel', 'GCA_Element', 'AGCB_Element', 'AGCB_Patch', 'CPM']


class NonLocalBlock(nn.Module):
    def __init__(self, planes, reduce_ratio=8):
        super(NonLocalBlock, self).__init__()

        inter_planes = planes // reduce_ratio
        self.query_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.key_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.value_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        proj_query = proj_query.contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = proj_key.contiguous().view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = proj_value.contiguous().view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma * out + x
        return out


class GCA_Channel(nn.Module):
    def __init__(self, planes, scale, reduce_ratio_nl, att_mode='origin'):
        super(GCA_Channel, self).__init__()
        assert att_mode in ['origin', 'post']

        self.att_mode = att_mode
        if att_mode == 'origin':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
            self.sigmoid = nn.Sigmoid()
        elif att_mode == 'post':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=1)
            self.conv_att = nn.Sequential(
                nn.Conv2d(planes, planes // 4, kernel_size=1),
                nn.BatchNorm2d(planes // 4),
                nn.ReLU(True),

                nn.Conv2d(planes // 4, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
                nn.Sigmoid(),
            )
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.att_mode == 'origin':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.sigmoid(gca)
        elif self.att_mode == 'post':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.conv_att(gca)
        else:
            raise NotImplementedError
        return gca


class GCA_Element(nn.Module):
    def __init__(self, planes, scale, reduce_ratio_nl, att_mode='origin'):
        super(GCA_Element, self).__init__()
        assert att_mode in ['origin', 'post']

        self.att_mode = att_mode
        if att_mode == 'origin':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
            self.sigmoid = nn.Sigmoid()
        elif att_mode == 'post':
            self.pool = nn.AdaptiveMaxPool2d(scale)
            self.non_local_att = NonLocalBlock(planes, reduce_ratio=1)
            self.conv_att = nn.Sequential(
                nn.Conv2d(planes, planes // 4, kernel_size=1),
                nn.BatchNorm2d(planes // 4),
                nn.ReLU(True),

                nn.Conv2d(planes // 4, planes, kernel_size=1),
                nn.BatchNorm2d(planes),
            )
            self.sigmoid = nn.Sigmoid()
        else:
            raise NotImplementedError

    def forward(self, x):
        batch_size, C, height, width = x.size()

        if self.att_mode == 'origin':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = F.interpolate(gca, [height, width], mode='bilinear', align_corners=True)
            gca = self.sigmoid(gca)
        elif self.att_mode == 'post':
            gca = self.pool(x)
            gca = self.non_local_att(gca)
            gca = self.conv_att(gca)
            gca = F.interpolate(gca, [height, width], mode='bilinear', align_corners=True)
            gca = self.sigmoid(gca)
        else:
            raise NotImplementedError
        return gca


class AGCB_Patch(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32, att_mode='origin'):
        super(AGCB_Patch, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)
        self.attention = GCA_Channel(planes, scale, reduce_ratio_nl, att_mode=att_mode)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## long context
        gca = self.attention(x)

        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            attention = gca[:, :, attention_ind[i], attention_ind[i+1]].view(batch_size, C, 1, 1)
            context_list.append(self.non_local(block) * attention)

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class AGCB_Element(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32, att_mode='origin'):
        super(AGCB_Element, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)
        self.attention = GCA_Element(planes, scale, reduce_ratio_nl, att_mode=att_mode)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## long context
        gca = self.attention(x)

        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            # attention = gca[:, :, attention_ind[i], attention_ind[i+1]].view(batch_size, C, 1, 1)
            context_list.append(self.non_local(block))

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = context * gca
        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class AGCB_NoGCA(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32):
        super(AGCB_NoGCA, self).__init__()

        self.scale = scale
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.conv = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ## single scale non local
        batch_size, C, height, width = x.size()

        local_x, local_y, attention_ind = [], [], []
        step_h, step_w = height // self.scale, width // self.scale
        for i in range(self.scale):
            for j in range(self.scale):
                start_x, start_y = i * step_h, j * step_w
                end_x, end_y = min(start_x + step_h, height), min(start_y + step_w, width)
                if i == (self.scale - 1):
                    end_x = height
                if j == (self.scale - 1):
                    end_y = width

                local_x += [start_x, end_x]
                local_y += [start_y, end_y]
                attention_ind += [i, j]

        index_cnt = 2 * self.scale * self.scale
        assert len(local_x) == index_cnt

        context_list = []
        for i in range(0, index_cnt, 2):
            block = x[:, :, local_x[i]:local_x[i+1], local_y[i]:local_y[i+1]]
            context_list.append(self.non_local(block))

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = self.conv(context)
        context = self.gamma * context + x
        context = self.relu(context)
        return context


class CPM(nn.Module):
    def __init__(self, planes, block_type, scales=(3,5,6,10), reduce_ratios=(4,8), att_mode='origin'):
        super(CPM, self).__init__()
        assert block_type in ['patch', 'element']
        assert att_mode in ['origin', 'post']

        inter_planes = planes // reduce_ratios[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(planes, inter_planes, kernel_size=1),
            nn.BatchNorm2d(inter_planes),
            nn.ReLU(True),
        )

        if block_type == 'patch':
            self.scale_list = nn.ModuleList(
                [AGCB_Patch(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
        elif block_type == 'element':
            self.scale_list = nn.ModuleList(
                [AGCB_Element(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)
                 for scale in scales])
        else:
            raise NotImplementedError

        channels = inter_planes * (len(scales) + 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, planes, 1),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        reduced = self.conv1(x)

        blocks = []
        for i in range(len(self.scale_list)):
            blocks.append(self.scale_list[i](reduced))
        out = torch.cat(blocks, 1)
        out = torch.cat((reduced, out), 1)
        out = self.conv2(out)
        return out