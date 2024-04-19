import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.YG2Model.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            in_inplanes = 256
        else:
            raise NotImplementedError

        self.decoder4 = DecoderBlock(in_inplanes, 256, BatchNorm)
        self.decoder3 = DecoderBlock(512, 128, BatchNorm)
        self.decoder2 = DecoderBlock(256, 64, BatchNorm, inp=True)
        self.decoder1 = DecoderBlock(128, 64, BatchNorm, inp=True)

        self.conv_e3 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())

        self.conv_e2 = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False),
                                     BatchNorm(128),
                                     nn.ReLU())

        self.conv_e1 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False),
                                     BatchNorm(64),
                                     nn.ReLU())

        self.sigmod = nn.Sigmoid()
        self.lay2o4 = nn.Conv2d(512,1,1)
        self.lay2o3 = nn.Conv2d(256,1,1)
        self.lay2o2 = nn.Conv2d(128,1,1)
        self.lay2o1 = nn.Conv2d(64,1,1)

        self._init_weight()


    def forward(self, e1, e2, e3, e4):
        lay2list = []
        d4 = torch.cat((self.decoder4(e4), self.conv_e3(e3)), dim=1)
        # print("d4:"+str(d4.shape))  # d4:torch.Size([5, 512, 16, 16])
        d3 = torch.cat((F.interpolate(self.decoder3(d4), scale_factor=2, mode='bilinear', align_corners=True), self.conv_e2(e2)), dim=1)
        # print("d3:"+str(d3.shape))  # d3:torch.Size([5, 256, 32, 32])
        d2 = torch.cat((self.decoder2(d3), self.conv_e1(e1)), dim=1)
        # print("d2:"+str(d2.shape))  # d2:torch.Size([5, 128, 64, 64])
        d1 = self.decoder1(d2)
        # print("d1:"+str(d1.shape))  # d1:torch.Size([5, 64, 128, 128])
        x = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)

        lay2o4 = self.sigmod(self.lay2o4(d4)) # 1,16,16
        lay2o3 = self.sigmod(self.lay2o3(d3)) # 1,32,32
        lay2o2 = self.sigmod(self.lay2o2(d2)) # 1,64,64
        lay2o1 = self.sigmod(self.lay2o1(d1)) # 1,128,128
        lay2list.append(lay2o4)
        lay2list.append(lay2o3)
        lay2list.append(lay2o2)
        lay2list.append(lay2o1)

        return x, lay2list

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone2, BatchNorm):
    return Decoder(num_classes, backbone2, BatchNorm)