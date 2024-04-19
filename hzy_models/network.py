import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import *
from .context import SCM
from .fusion import *
from models.YG1_hzy.hzy_models.MDPM import MDPM
from models.YG1_hzy.do_conv_pytorch import DOConv2d
from models.YG1_hzy.laplace import make_laplace_pyramid
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rgb_to_grayscale
import os

__all__ = ['mfdsnet']


class _FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, drop=0.5):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            # nn.Conv2d(in_channels, inter_channels, 3, 1, 1),  # (32,32)
            DOConv2d(in_channels, inter_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(drop),
            # nn.Conv2d(inter_channels, out_channels, 1, 1, 0)  # (32,1)
            DOConv2d(inter_channels, out_channels, 1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.block(x)


class MFDS_Net(nn.Module):
    def __init__(self, backbone='resnet34', scales=(10, 6), reduce_ratios=(8, 8), gca_type='patch', gca_att='origin',
                 drop=0.1, switch_backbone2=False, backbone2='resnet18', output_stride=16, num_classes=21, num_neighbor=9, sync_bn=True,
                 freeze_bn=False):
        super(MFDS_Net, self).__init__()
        assert backbone in ['resnet18', 'resnet34', 'resnet50']
        assert backbone2 in ['resnet18', 'resnet34', 'resnet50']
        assert gca_type in ['patch', 'element']
        assert gca_att in ['origin', 'post']

        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True)
        elif backbone == 'resnet34':
            self.backbone = resnet34(pretrained=True)
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True)
        else:
            raise NotImplementedError

        if switch_backbone2:
            if backbone2 == 'resnet18':
                self.backbone2 = resnet18(pretrained=True)
            elif backbone2 == 'resnet34':
                self.backbone2 = resnet34(pretrained=True)
            elif backbone2 == 'resnet50':
                self.backbone2 = resnet50(pretrained=True)
            else:
                raise NotImplementedError

        self.fuse23 = DFIM(512, 256, 256)
        self.fuse12 = DFIM(256, 128, 128)

        self.head = _FCNHead(128, 2, drop=drop)

        self.context = SCM(planes=512, scales=scales, reduce_ratios=reduce_ratios, block_type=gca_type,
                           att_mode=gca_att)


        self.mdpm_1 = MDPM(128, 128)
        self.mdpm_2 = MDPM(256, 256)
        self.mdpm_3 = MDPM(512, 512)



        self.weight = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(512, 512 // 4, 1, 1, 0),
            DOConv2d(512, 512 // 4, 1, stride=1, padding=0),
            nn.BatchNorm2d(512 // 4),
            nn.ReLU(),

            # nn.Conv2d(512 // 4, 512, 1, 1, 0),
            DOConv2d(512 // 4, 512, 1, stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.Sigmoid()
        )
        self.fuse_conv = nn.Sequential(
            # nn.Conv2d(1024, 512, kernel_size=1, bias=False),
            DOConv2d(1024, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.skip2_conv = nn.Sequential(
            # nn.Conv2d(512, 256, kernel_size=1, bias=False),
            DOConv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.skip1_conv = nn.Sequential(
            # nn.Conv2d(256, 128, kernel_size=1, bias=False),
            DOConv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.num_images = 0
        self.sigmoid = nn.Sigmoid()

        # =-------------------------------------------------


        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.sigmoid = nn.Sigmoid()
        self.lay1o1 = DOConv2d(512,2,1)
        self.lay1o2 = DOConv2d(256,2,1)
        self.lay1o3 = DOConv2d(128,2,1)

    def vis_am(self, am, img_name):
        path = "./feas"
        os.makedirs(path,exist_ok=True)
        for i in range(len(img_name)):
            for j, f in enumerate(am):
                f =f[0].cpu().mean(dim=0)
                path_img = f'{img_name[i]}_{j}.png'
                save_path = os.path.join(path, path_img)
                fig, ax = plt.subplots(1, 1, tight_layout=True)
                f = f.detach().numpy()
                ax.imshow(f, cmap='jet')
                # plt.show()
                ax.axis('off')
                plt.savefig(save_path, dpi=200, bbox_inches='tight')
                plt.close()


    def forward(self, x, y, img_name):

        grayscale_img_x = rgb_to_grayscale(x) 
        edge_feature_x = make_laplace_pyramid(grayscale_img_x, 5, 1)
        edge_feature_x = edge_feature_x[1]

        grayscale_img_y = rgb_to_grayscale(y) 
        edge_feature_y = make_laplace_pyramid(grayscale_img_y, 5, 1)
        edge_feature_y = edge_feature_y[1]

        
        lay1list = []
        lay2list = []
        _, _, hei, wid = x.shape
        c1, c2, c3 = self.backbone(x)  # Layer1 Layer2 Layer3
        a1, a2, a3 = self.backbone(y)

        # c1:[4, 128, 128, 128]
        # c2:[4, 256, 64, 64]
        # c3:[5, 512, 32, 32]
        
        c1rfb ,am_x1= self.mdpm_1(c1, edge_feature_x)  # 128,128,128
        c2rfb ,am_x2= self.mdpm_2(c2, edge_feature_x)  # 256,64,64
        c3rfb ,am_x3= self.mdpm_3(c3, edge_feature_x)  # 512,32,32

        lossc3 = self.lay1o1(c3rfb)  # 32,32
        lay1list.append(lossc3)

        # hotmaplist.append(c1)
        # hotmaplist.append(c2)
        # hotmaplist.append(c3)
        # hotmaplist.append(c1rfb)
        # hotmaplist.append(c2rfb)
        # hotmaplist.append(c3rfb)

        a1rfb, am_y1 = self.mdpm_1(a1, edge_feature_y)
        a2rfb, am_y2 = self.mdpm_2(a2, edge_feature_y)
        a3rfb, am_y3 = self.mdpm_3(a3, edge_feature_y)

        # am=[am_x1, am_y1]
        
        # self.vis_am(am, img_name)

        # hotmaplist.append(a1)
        # hotmaplist.append(a2)
        # hotmaplist.append(a3)
        # hotmaplist.append(a1rfb)
        # hotmaplist.append(a2rfb)
        # hotmaplist.append(a3rfb)

        lossa3 = self.lay1o1(a3rfb)  # 32，32
        lay2list.append(lossa3)

        c3_weight = self.weight(c3rfb)  # 512,1,1
        a3_weight = self.weight(a3rfb)  # 512,1,1


        c3rfb = c3rfb * c3_weight
        a3rfb = a3rfb * a3_weight

        # hotmaplist.append(c3rfb)
        # hotmaplist.append(a3rfb)

        out = self.context(c3rfb)  
        aout = self.context(a3rfb)

        # hotmaplist.append(out)
        # hotmaplist.append(aout)

        totalout = torch.cat((out, aout), dim=1)
        totalout = self.fuse_conv(totalout)  # 512,64,64

        skip2 = torch.cat((c2rfb, a2rfb), dim=1)  
        skip2 = self.skip2_conv(skip2)

        # hotmaplist.append(skip2)  # hotmap

        totalout = F.interpolate(totalout, size=[hei // 4, wid // 4], mode='bilinear', align_corners=True)  # up sample


        totalout = self.fuse23(totalout, skip2)
        # hotmaplist.append(totalout)  # hotmap


        lossout1 = self.lay1o2(totalout)
        lay1list.append(lossout1)
        lossaout1 = self.lay1o2(totalout)
        lay2list.append(lossaout1)


        skip1 = torch.cat((c1rfb, a1rfb), dim=1)  
        skip1 = self.skip1_conv(skip1)
        # skip1 = self.iaff1(c1rfb, a1rfb)
        # hotmaplist.append(skip1)  # hotmap

        totalout = F.interpolate(totalout, size=[hei // 2, wid // 2], mode='bilinear', align_corners=True)  # up samlpe


        totalout = self.fuse12(totalout, skip1)  # AFM  # 128,128,128
        # hotmaplist.append(totalout)  # hotmap


        lossout2 = self.lay1o3(totalout)
        lay1list.append(lossout2)
        lossaout2 = self.lay1o3(totalout)
        lay2list.append(lossaout2)

        pred = self.head(totalout)  # Conv + batchNorm + Relu
        out = F.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)  # up sample
        # hotmaplist.append(out)  # hotmap
        out = self.sigmoid(out)
        # hotmaplist.append(out)  # hotmap

        outlist = []
        outlist.append(out)

        return outlist, lay1list, lay2list

def mfdsnet(backbone, scales, reduce_ratios, gca_type, gca_att, drop, switch_backbone2, backbone2, output_stride, num_classes,
            num_neighbor, sync_bn, freeze_bn):
    return MFDS_Net(backbone=backbone, scales=scales, reduce_ratios=reduce_ratios, gca_type=gca_type, gca_att=gca_att,
                   drop=drop, switch_backbone2=switch_backbone2, backbone2=backbone2, output_stride=output_stride, num_classes=num_classes,
                   num_neighbor=num_neighbor, sync_bn=sync_bn, freeze_bn=freeze_bn)
