import torch
from thop import profile


def FLOPs_and_Params(model, min_size, max_size, device):
    x = torch.randn(1, 3, min_size, max_size).to(device)
    bmask = torch.randn(1, min_size, max_size).to(device)
    model.trainable = False
    model.eval()

    print('==============================')
    flops, params = profile(model, inputs=(x, bmask))
    print('==============================')
    print('FLOPs : {:.2f} B'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))

    model.trainable = True
    model.train()


if __name__ == "__main__":
    pass
