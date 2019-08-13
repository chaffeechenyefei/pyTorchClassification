from __future__ import print_function, division, absolute_import
from functools import partial

import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as M
import pretrainedmodels
import math
import torch.utils.model_zoo as model_zoo

from .xception import *
from .inceptionv4 import *
from .inceptionresnetv2 import *
from .dpn import *

from models.dictLayer import DictLayer

__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d']
from collections import OrderedDict

#===================================================================================================
#===================================================================================================
#basic models
#===================================================================================================
#===================================================================================================
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])

# #import from others github
# class BasicConv2d(nn.Module):

#     def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
#         super(BasicConv2d, self).__init__()
#         self.conv = nn.Conv2d(in_planes, out_planes,
#                               kernel_size=kernel_size, stride=stride,
#                               padding=padding, bias=False) # verify bias false
#         self.bn = nn.BatchNorm2d(out_planes,
#                                  eps=0.001, # value found in tensorflow
#                                  momentum=0.1, # default pytorch value
#                                  affine=True)
#         self.relu = nn.ReLU(inplace=True)

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x

# #Example of mix two branch into one output
# class Mixed_4a(nn.Module):

#     def __init__(self):
#         super(Mixed_4a, self).__init__()

#         self.branch0 = nn.Sequential(
#             BasicConv2d(160, 64, kernel_size=1, stride=1),
#             BasicConv2d(64, 96, kernel_size=3, stride=1)
#         )

#         self.branch1 = nn.Sequential(
#             BasicConv2d(160, 64, kernel_size=1, stride=1),
#             BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
#             BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
#             BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
#         )

#     def forward(self, x):
#         x0 = self.branch0(x)
#         x1 = self.branch1(x)
#         out = torch.cat((x0, x1), 1) #Nx{[C]_[cat]}xHxW
#         return out

#===================================================================================================
#===================================================================================================
#===================================================================================================
def create_net(net_cls, pretrained: bool):
    # if ON_KAGGLE and pretrained:
    #     net = net_cls()
    #     model_name = net_cls.__name__
    #     weights_path = f'../input/{model_name}/{model_name}.pth'
    #     net.load_state_dict(torch.load(weights_path))
    # else:
    net = net_cls(pretrained=pretrained)
    return net


#===================================================================================================
#===================================================================================================
# ResNet and DenseNet
#===================================================================================================
#===================================================================================================
class ResNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet50, dropout=False):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    #freeze param
    def freeze(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


#my_model = nn.Sequential(*list(pretrained_model.children())[:-1])
#resnet+1024fc+128fc+softmax
class ResNetV2(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet50, dropout=False,fc_dim = 1024):
        super(ResNetV2,self).__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Linear(self.net.fc.in_features, fc_dim),
                nn.Dropout(),
                nn.ReLU(),
            )
        else:
            self.net.fc = nn.Sequential(
                nn.Linear(self.net.fc.in_features, fc_dim),
                nn.ReLU(),
                )
            

        self.last_fc = nn.Linear(fc_dim, num_classes)

    #freeze param
    def freeze(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        fc1 = self.net(x)
        fc2 = self.last_fc(fc1)
        return fc1,fc2


class ResNetV3(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet50, dropout=False,fc_dim_per_class = 10):
        super(ResNetV3,self).__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()

        if dropout:
            self.net.fc = nn.Sequential(
                nn.Linear(self.net.fc.in_features, 1024),
                nn.Dropout(),
                nn.ReLU(),
            )
        else:
            self.net.fc = nn.Sequential(
                nn.Linear(self.net.fc.in_features, 1024),
                nn.ReLU(),
                )

        self.dict_layer = DictLayer( 1024,  fc_dim_per_class*num_classes , num_classes,alpha=0.1)
        self.last_layer = nn.Linear(fc_dim_per_class*num_classes, num_classes)

    #freeze param
    def freeze(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x, targets):
        fc1 = self.net(x)
        fc1 = self.dict_layer(fc1,targets)
        fc2 = self.last_layer(fc1)
        loss = self.dict_layer.getLoss()
        return fc1,fc2,loss

class ResNetV4(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet50, dropout=False):
        super(ResNetV4,self).__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()

        if dropout:
            self.net.fc = nn.Sequential(
                nn.Linear(self.net.fc.in_features, 1024),
                nn.Dropout(),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes),
                )
        else:
            self.net.fc = nn.Sequential(
                nn.Linear(self.net.fc.in_features, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, num_classes),
                )

    #freeze param
    def freeze(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        fc1 = self.net(x)
        return fc1

class DenseNet(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.densenet121):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.avg_pool = AvgPool()
        self.net.classifier = nn.Linear(
            self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out



"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""


pretrained_settings = {
    'senet154': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet50': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet101': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnet152': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext50_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = int(math.floor(planes * (base_width / 64)) * groups)


        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )

        self.avg_pool = AvgPool()
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)


    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model(model, num_classes, settings):
    # assert num_classes == settings['num_classes'], \
    #     'num_classes should be {}, but is {}'.format(
    #         settings['num_classes'], num_classes)

    pretrained_dict = model_zoo.load_url(settings['url'])
    model_dict = model.state_dict()

    # 将pretrained_dict里不属于model_dict的键剔除掉
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'last_linear' not in k}
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 更新现有的model_dict
    model_dict.update(pretrained_dict)
    # 加载我们真正需要的state_dict
    model.load_state_dict(model_dict)

    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']


def senet154(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
                  dropout_p=0.2, num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['senet154'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet50(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet50'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet101(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet101'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnet152(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnet152'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)

    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


class SE50MSC(nn.Module):

    def __init__(self, num_classes=1000, pretrained='imagenet'):
        super(SE50MSC, self).__init__()
        self.basemodel = se_resnext50_32x4d(num_classes=num_classes, pretrained='imagenet')
        self.basemodel.layer0.conv1 = nn.Sequential()
        self.basemodel.layer0.bn1 = nn.Sequential()

        self.MSC_5x5 = nn.Conv2d(3, 32, 5, stride=2, padding=2, bias=False)
        self.MSC_7x7 = nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False)
        self.MSC_9x9 = nn.Conv2d(3, 32, 9, stride=2, padding=4, bias=False)
        self.MSC_bn = nn.BatchNorm2d(96)

        self.MSC_conv1x1 = nn.Conv2d(96, 64, 1, stride=1, bias=False)
        self.MSC_conv1x1_bn = nn.BatchNorm2d(64)

    def forward(self, x):
        MSC_5x5 = self.MSC_5x5(x)
        MSC_7x7 = self.MSC_7x7(x)
        MSC_9x9 = self.MSC_9x9(x)
        x = torch.cat([MSC_5x5,MSC_7x7,MSC_9x9],dim=1)

        x = F.relu(self.MSC_bn(x))
        x = self.MSC_conv1x1(x)
        x = F.relu(self.MSC_conv1x1_bn(x))
        x = self.basemodel(x)
        return x


se_resnext50_msc = partial(SE50MSC)
se_resnext50 = partial(se_resnext50_32x4d)
se_resnext101 = partial(se_resnext101_32x4d)
se_resnet152 = partial(se_resnet152)
inception_v4 = partial(inceptionv4)
inceptionresnet_v2 = partial(inceptionresnetv2)
dpn_92 = partial(dpn92)
dpn_68b = partial(dpn68b)

resnet18 = partial(ResNet, net_cls=M.resnet18)
resnet34 = partial(ResNet, net_cls=M.resnet34)
resnet50 = partial(ResNet, net_cls=M.resnet50)
resnet50V2 = partial(ResNetV2, net_cls=M.resnet50)
resnet50V3 = partial(ResNetV3, net_cls=M.resnet50)
resnet50V4 = partial(ResNetV4, net_cls=M.resnet50)
resnet101 = partial(ResNet, net_cls=M.resnet101)
resnet152 = partial(ResNet, net_cls=M.resnet152)

densenet121 = partial(DenseNet, net_cls=M.densenet121)
densenet169 = partial(DenseNet, net_cls=M.densenet169)
densenet201 = partial(DenseNet, net_cls=M.densenet201)
densenet161 = partial(DenseNet, net_cls=M.densenet161)


