from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torchvision.models as models
from torchvision.models import _utils as det_utils

''' Taken from https://github.com/pochih/FCN-pytorch
    The net is named as "FCN8s" but in fact has a Encoder-Decoder structure with skip layers between 
    both elements. A VGG16 (default) is used for feature extraction.
    Input has to be dividable by 32
'''

class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)

        return score # size=(N, n_class, x.H/1, x.W/1)
    

class ResNet(nn.Module):

    def __init__(self, resnet):
        super().__init__()
        self.resnet = resnet
        self.intermediate = det_utils.IntermediateLayerGetter(self.resnet, {"layer4" : "x5", "layer3" : "x4",
                                                                               "layer2" : "x3", "layer1" : "x2"})

    def forward(self, x):
        inter_dict = self.intermediate(x)
        return inter_dict


if __name__ == "__main__":

    batch_size, n_class, h, w = 10, 20, 160, 160

    # test output size
    resnet18 = models.resnet34(pretrained=True)
    res_model = ResNet(resnet18)
    model = FCN8s(res_model, 2)

    input = torch.autograd.Variable(torch.randn(batch_size, 3, 960, 960))
    output = model(input)