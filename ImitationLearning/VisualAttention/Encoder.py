import torch
import torch.nn as nn
import torch.nn.functional as F
from   torchvision import models

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
    # https://discuss.pytorch.org/t/break-resnet-into-two-parts/39315

    expansion: int = 4

    def __init__( self,
                  inplanes: int,      # d_in
                  planes: int,        # n_hidden, dhdn
                  stride: int = 1,
                  groups: int = 1,
                  base_width: int = 64,
                  dilation: int = 1):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size =     1, 
                                                bias        = False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size =        3, 
                                             stride      =   stride, 
                                             padding     = dilation, 
                                             groups      =   groups, 
                                             bias        =    False, 
                                             dilation    = dilation)
        self.bn2 = nn.BatchNorm2d(width)
        
        self.conv3 = nn.Conv2d(width, planes*self.expansion, kernel_size =     1, 
                                                             bias        = False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if stride != 1 or inplanes != planes*self.expansion:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes*self.expansion, 
                                                                kernel_size =      1, 
                                                                stride      = stride, 
                                                                bias        = False),
                                            nn.BatchNorm2d(planes * self.expansion))
        else:
            self.downsample = None
        

    def forward(self,x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class HighResNet34(nn.Module):
    def __init__(self):
        super(HighResNet34, self).__init__()
        # [3, 4 | 6, 3]
        self.layer3a = Bottleneck(128, 64, stride=2)    # 1
        self.layer3b = Bottleneck(256, 64, stride=1)    # 2
        self.layer3c = Bottleneck(256, 64, stride=1)    # 3
        self.layer3d = Bottleneck(256, 64, stride=1)    # 4
        self.layer3e = Bottleneck(256, 64, stride=1)    # 5
        self.layer3f = Bottleneck(256, 64, stride=1)    # 6

        self.layer4a = Bottleneck(256,128, stride=2)    # 1
        self.layer4b = Bottleneck(512,128, stride=1)    # 2
        self.layer4c = Bottleneck(512,128, stride=1)    # 3

    def forward(self, x):
        x = self.layer3a(x)
        x = self.layer3b(x)
        x = self.layer3c(x)
        x = self.layer3d(x)
        x = self.layer3e(x)
        x = self.layer3f(x)

        x = self.layer4a(x)
        x = self.layer4b(x)
        x = self.layer4c(x)
        
        return x


class HighResNet50(nn.Module):
    def __init__(self):
        super(HighResNet50, self).__init__()
        # [3, 4 | 6, 3]
        self.layer3a = Bottleneck( 512, 256, stride=2)    # 1
        self.layer3b = Bottleneck(1024, 256, stride=1)    # 2
        self.layer3c = Bottleneck(1024, 256, stride=1)    # 3
        self.layer3d = Bottleneck(1024, 256, stride=1)    # 4
        self.layer3e = Bottleneck(1024, 256, stride=1)    # 5
        self.layer3f = Bottleneck(1024, 256, stride=1)    # 6

        self.layer4a = Bottleneck(1024, 512, stride=2)    # 1
        self.layer4b = Bottleneck(2048, 512, stride=1)    # 2
        self.layer4c = Bottleneck(2048, 512, stride=1)    # 3

    def forward(self, x):
        x = self.layer3a(x)
        x = self.layer3b(x)
        x = self.layer3c(x)
        x = self.layer3d(x)
        x = self.layer3e(x)
        x = self.layer3f(x)

        x = self.layer4a(x)
        x = self.layer4b(x)
        x = self.layer4c(x)
        
        return x
        
