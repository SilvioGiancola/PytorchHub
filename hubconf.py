from torchvision.models.resnet import resnet18 as _resnet18
dependencies = ['torch']

# resnet18 is the name of entrypoint


def resnet18(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _resnet18(pretrained=pretrained, **kwargs)
    return model
