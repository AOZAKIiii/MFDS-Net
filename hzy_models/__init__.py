from .network import *
from .context import *


def YG1get_segmentation_model(name):
    if name == 'hzy':
        net = mfdsnet(backbone='resnet34', scales=(10, 6, 5, 3), reduce_ratios=(16, 4), gca_type='patch', gca_att='post',
                      drop=0.1, switch_backbone2=False, backbone2='resnet18', output_stride=16, num_classes=21, num_neighbor=9, sync_bn=True, freeze_bn=False)
    elif name == 'agpcnet_2':
        net = mfdsnet(backbone='resnet18', scales=(10, 6, 5, 4), reduce_ratios=(16, 4), gca_type='patch', gca_att='post', drop=0.1)
    else:
        raise NotImplementedError

    return net
