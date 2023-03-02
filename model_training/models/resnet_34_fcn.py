from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.models as models
from torchvision.models import _utils as det_utils


class FCNHead(nn.Sequential):
    def __init__(self, in_channels: int, channels: int) -> None:
        inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(),
            nn.Conv2d(inter_channels, channels, 1),
        ]

        super().__init__(*layers)


class ResNetFCN(nn.Module):    
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.intermediate = det_utils.IntermediateLayerGetter(self.pretrained_net, {"layer4" : "x5"})
        self.classifier = FCNHead(512, n_class)


    def forward(self, x):
        input_shape = x.shape[-2:]
        output = self.intermediate(x)
        output = output["x5"]

        x = self.classifier(output)
        x = torch.nn.functional.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x



if __name__ == "__main__":
    
    batch_size, n_class, h, w = 4, 20, 960, 640

    # test output size
    resnet34 = models.resnet34(pretrained=True)
    model = ResNetFCN(resnet34, 2)

    input = torch.autograd.Variable(torch.randn(batch_size, 3, 960, 960))
    output = model(input)
    pass