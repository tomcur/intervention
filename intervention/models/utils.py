from torch import nn


def freeze_batch_norm_layers(m: nn.Module):
    """
    Freezes all batch normalization layers in module `m`.

    The batch normalization layers' parameters are set to not require gradients, and the
    module is set to evaluation mode such that running statistics from previous training
    are used for normalization.

    This should be called every time after `.train()` has been called on `m` (or any of
    its submodules).
    """
    for module in m.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
