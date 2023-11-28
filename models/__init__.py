import torch
from .snpmiso.build import build_snpmiso



# build object detector
def build_model(args, 
                cfg, 
                device, 
                num_classes=80, 
                trainable=False, 
                pretrained=None,
                eval_mode=False):
    print('==============================')
    print('Build {} ...'.format(args.version.upper()))

    if args.version in ['snpmiso-r50', 'snpmiso-r101']:
        return build_snpmiso(args, cfg, device, num_classes, trainable, pretrained, eval_mode)
