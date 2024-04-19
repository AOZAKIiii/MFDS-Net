import torch
import torch.nn as nn
import torch.nn.functional as F
from models.YG1_hzy.do_conv_pytorch import DOConv2d


__all__ = ['NonLocalBlock', 'G_Channel_atten', 'ChannelAttention', 'Atten_Patch', 'SCM']


class NonLocalBlock(nn.Module):
    def __init__(self, planes, reduce_ratio=8):
        super(NonLocalBlock, self).__init__()

        inter_planes = planes // reduce_ratio
        self.query_conv = DOConv2d(planes, inter_planes, kernel_size=1)
        self.key_conv = DOConv2d(planes, inter_planes, kernel_size=1)
        self.value_conv = DOConv2d(planes, planes, kernel_size=1)
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


class G_Channel_atten(nn.Module):
    def __init__(self, planes, scale, reduce_ratio_nl, att_mode='origin'):
        super(G_Channel_atten, self).__init__()
        assert att_mode in ['origin', 'post']

        self.pool = nn.AdaptiveMaxPool2d(scale)
        self.non_local_att = NonLocalBlock(planes, reduce_ratio=1)
        self.conv_att = nn.Sequential(
            DOConv2d(planes, planes // 4, kernel_size=1),
            # nn.Conv2d(planes, planes // 4, kernel_size=1),
            nn.BatchNorm2d(planes // 4),
            nn.ReLU(True),

            DOConv2d(planes // 4, planes, kernel_size=1),
            # nn.Conv2d(planes // 4, planes, kernel_size=1),
            nn.BatchNorm2d(planes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        
        gca = self.pool(x)
        gca = self.non_local_att(gca)
        gca = self.conv_att(gca)

        return gca


class Atten_Patch(nn.Module):
    def __init__(self, planes, scale=2, reduce_ratio_nl=32, att_mode='origin'):
        super(Atten_Patch, self).__init__()

        self.scale = scale
        # self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.local= LocalContextModule(planes, reduce_ratio_nl=reduce_ratio_nl)

        self.conv = nn.Sequential(
            DOConv2d(planes, planes, 3, stride=1, padding=1),
            # nn.Conv2d(planes, planes, 3, 1, 1),
            nn.BatchNorm2d(planes),
            # nn.Dropout(0.1)
        )
        self.relu = nn.ReLU(True)
        self.attention = G_Channel_atten(planes, scale, reduce_ratio_nl, att_mode=att_mode)
        self.w = nn.Parameter(torch.zeros(1))

        self.gamma = nn.Parameter(torch.zeros(1))
        
        # self.channel_atten = ChannelAttention(planes, reduction_ratio=16)

    def forward(self, x):
        ## long context
        res = x
        # x = x + self.channel_atten(x) * self.w
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
            # context_list.append(self.non_local(block) * attention)
            context_list.append(self.local(block) * attention)

        tmp = []
        for i in range(self.scale):
            row_tmp = []
            for j in range(self.scale):
                row_tmp.append(context_list[j + i * self.scale])
            tmp.append(torch.cat(row_tmp, 3))
        context = torch.cat(tmp, 2)

        context = self.conv(context)

        # context = self.gamma * context + x
        context = self.gamma * context + res
        # context = context + x
        #
        context = self.relu(context)
        return context

# backbone='resnet18', scales=(10, 6, 5, 3), reduce_ratios=(16, 4), gca_type='patch', gca_att='post', drop=0.1
class SCM(nn.Module):  # planes:512
    def __init__(self, planes, block_type, scales=(3,5,6,10), reduce_ratios=(4,8), att_mode='origin'):
        super(SCM, self).__init__()
        assert block_type in ['patch', 'element']
        assert att_mode in ['origin', 'post']

        inter_planes = planes // reduce_ratios[0]  # 32
        self.conv1 = nn.Sequential(
            DOConv2d(planes, inter_planes, kernel_size=1),
            # nn.Conv2d(planes, inter_planes, kernel_size=1),
            nn.BatchNorm2d(inter_planes),
            nn.ReLU(True),
        )

        # scale: 10,6,5,3  reduce_ratios: 4
        self.scale_list = nn.ModuleList([Atten_Patch(inter_planes, scale=scale, reduce_ratio_nl=reduce_ratios[1], att_mode=att_mode)  # scale: 10,6,5,3
                                        for scale in scales])

        channels = inter_planes * (len(scales) + 1)

        self.conv2 = nn.Sequential(
            DOConv2d(channels, planes, 1),
            # nn.Conv2d(channels, planes, 1),
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



class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return out * x
    

class LocalContextModule(nn.Module):
    def __init__(self, planes, reduce_ratio_nl):
        super(LocalContextModule, self).__init__()
        self.conv = nn.Sequential(
            DOConv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(planes),
            # nn.ReLU(True)
        )
        self.non_local = NonLocalBlock(planes, reduce_ratio=reduce_ratio_nl)
        self.w = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        context = self.conv(x)
        context = self.non_local(context)
        return self.w * context + x