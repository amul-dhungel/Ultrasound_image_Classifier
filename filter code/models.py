
from torch import nn,rand
from torchvision import models
from hyperparameters import parameters as params
import torch.random


def resnet18():
    """
        efficientnet ResNet 18 model definition.
    """
    out_features = params['out_features']

    model = models.resnet18(pretrained=True)

    # To freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # New output layers
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, out_features)

    return model


def efficientnet():
    """
        efficientnet EfficientNet B4 model definition.
    """
    out_features = params['out_features']

    model = models.efficientnet_b7(pretrained=True)

    model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=out_features)
    return model


def vision_transforer_vit():
    m = nn.Linear(20, 30)
    input = torch.randn(128, 20)
    output = m(input)
    print(output.size())

vision_transforer_vit()