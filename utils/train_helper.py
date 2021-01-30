import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from graphs.models.new_deeplab_multi import DeeplabMulti as NewDeeplabMulti
from graphs.models.deeplab_multi import DeeplabMulti
from graphs.models.deeplab_vgg import DeeplabVGG
from graphs.models.vgg_fcn8s import VGG16_FCN8s

def get_model(args):
    if args.backbone == "deeplabv2_multi":
        model = DeeplabMulti(num_classes=args.num_classes,
                             pretrained=args.imagenet_pretrained)
    elif args.backbone == "new_deeplabv2_multi":
        model = NewDeeplabMulti(num_classes=args.num_classes,
                             pretrained=args.imagenet_pretrained,
                             use_se=args.use_se,
                             train_bn=not args.freeze_bn,
                             norm_style=args.norm_style)
    elif args.backbone == 'deeplabv3_resnest50':
        from encoding.models import get_segmentation_model
        from encoding.nn import SyncBatchNorm
        model = get_segmentation_model('deeplab', dataset='citys', backbone='resnest50', aux=True, norm_layer=SyncBatchNorm)
    elif args.backbone == 'deeplabv3_resnest101':
        from encoding.models import get_segmentation_model
        from encoding.nn import SyncBatchNorm
        model = get_segmentation_model('deeplab', dataset='citys', backbone='resnest101', aux=True, norm_layer=SyncBatchNorm)
    elif args.backbone == 'deeplabv2_vgg':
        model = DeeplabVGG(num_classes=args.num_classes)
    elif args.backbone == 'vgg16_fcn8s':
        model = VGG16_FCN8s(num_classes=args.num_classes)
    elif args.backbone == 'hrnet':
        raise NotImplementedError()
        # from graphs.models.hrnet import HighResolutionNet
        # model = HighResolutionNet(cfg)
        # model.init_weights(self.args.pretrained_ckpt_file)
    else:
        raise NotImplementedError()

    if 'deeplabv2' in args.backbone or 'vgg16_fcn8s' == args.backbone:
        params = model.optim_parameters(args)
    else:
        # https://github.com/zhanghang1989/PyTorch-Encoding/blob/master/experiments/segmentation/train.py#L153
        params = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        if hasattr(model, 'head'):
            params.append({'params': model.head.parameters(), 'lr': args.lr*10})
            print("Model head has 10x LR")
        if hasattr(model, 'auxlayer'):
            params.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
            print("Model auxlayer has 10x LR")

    args.numpy_transform = True
    return model, params
