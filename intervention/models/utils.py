from torch import nn


def freeze_batch_norm_layers(m: nn.Module):
    for module in m.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
