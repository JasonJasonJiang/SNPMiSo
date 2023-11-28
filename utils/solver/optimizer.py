from torch import optim


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def build_optimizer(cfg, args, model, base_lr=0.0, backbone_lr=0.0):
    print('==============================')
    print('Optimizer: {}'.format(cfg['optimizer']))
    print('--momentum: {}'.format(cfg['momentum']))
    print('--weight_decay: {}'.format(cfg['weight_decay']))

    param_dicts = [
        {
            "params":
                [p for n, p in model.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": base_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": backbone_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": base_lr * args.lr_linear_proj_mult,
        }
    ]

    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            params=param_dicts, 
            lr=base_lr,
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay']
            )

    elif cfg['optimizer'] == 'adam':
        optimizer = optim.Adam(
            params=param_dicts, 
            lr=base_lr,
            weight_decay=cfg['weight_decay']
            )
                                
    elif cfg['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            params=param_dicts, 
            lr=base_lr,
            weight_decay=cfg['weight_decay']
            )
                                
    return optimizer
