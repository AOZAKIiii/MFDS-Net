from modeling.backbone import resnet


def build_backbone(backbone2, output_stride, BatchNorm):
    if backbone2 == 'resnet':
        return resnet.ResNet101(output_stride, BatchNorm)
    else:
        raise NotImplementedError
