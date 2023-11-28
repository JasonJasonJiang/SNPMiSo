from .snpmiso_config import snpmiso_config


def build_config(args):
    if args.version in ['snpmiso-r50', 'snpmiso-r101']:
        return snpmiso_config[args.version]
 
