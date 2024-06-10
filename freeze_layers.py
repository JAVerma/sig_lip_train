def freeze_layer(module, unfreezed_layer=4):
    for param in module.parameters():
        param.requires_grad_(False)
    for layer in list(module.children())[-unfreezed_layer:]:
        for param_l in layer.parameters():
            param_l.requires_grad_(True)
