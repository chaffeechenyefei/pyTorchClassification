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
from .maskLayer import maskLayer

from models.dictLayer import DictLayer
from models.netvlad import NetVladLayer,NetVladLayerV2

from utils import load_model_with_dict_replace,load_model_with_dict,load_model

__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152',
           'se_resnext50_32x4d', 'se_resnext101_32x4d','vggnetvlad','location_recommend_model_v1','location_recommend_model_v3']
from collections import OrderedDict

#===================================================================================================
#===================================================================================================
#basic models
#===================================================================================================
#===================================================================================================
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])

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
# VGGNet
#===================================================================================================
#===================================================================================================
class VggNetVLAD(nn.Module):
    def __init__(self, net_cls=M.vgg16, pretrained='imagenet', center_num = 64):
        super().__init__()
        basemodel = create_net( net_cls=net_cls, pretrained=pretrained)
        self.net = nn.Sequential(*list(basemodel.features.children())[:-2])
        self.vlad = NetVladLayerV2(num_clusters= center_num, dim = 512)
        self.featLen = center_num*512

    def forward(self, input):
        x = self.net(input)
        x = self.vlad(x)
        return x

    def finetune(self, model_path:str):
        load_model_with_dict(self.net, model_path, 'encoder.', '')
        load_model_with_dict(self.vlad, model_path, 'pool.', '')

    def freeze_net(self):
        for param in self.net.parameters():
            param.requires_grad = False

# ===================================================================================================
# ===================================================================================================
# Location Recommendation Model
# ===================================================================================================
# ===================================================================================================
class NaiveDL(nn.Module):
    """
    2 class classification model
    """
    def __init__(self,feat_comp_dim=102,feat_loc_dim=23):
        super().__init__()
        self._common_feat_dim = 64
        self._feat_comp_dim = feat_comp_dim
        self._feat_loc_dim = feat_loc_dim
        self.net_comp = nn.Sequential(
            nn.Linear(feat_comp_dim,256,bias=True),
            nn.LeakyReLU(),
            nn.Linear(256,128,bias=True),
            nn.LeakyReLU(),
            nn.Linear(128, self._common_feat_dim, bias=True),
            nn.LeakyReLU(),
        )

        self.net_loc = nn.Sequential(
            nn.Linear(feat_loc_dim,64,bias=True),
            nn.LeakyReLU(),
            nn.Linear(64,self._common_feat_dim,bias=True),
            nn.LeakyReLU(),
        )

        self.net_shared = nn.Sequential(
            nn.Linear(2*self._common_feat_dim,self._common_feat_dim,bias=True),
            nn.LeakyReLU(),
        )

        self.classifer = nn.Linear(self._common_feat_dim,2,bias=False)

    def forward(self, feat_comp,feat_loc):
        assert(feat_comp.shape[1]==self._feat_comp_dim)
        assert(feat_loc.shape[1] == self._feat_loc_dim)
        common_feat_comp = self.net_comp(feat_comp)
        common_feat_loc = self.net_loc(feat_loc)
        concat_feat = torch.cat([common_feat_comp,common_feat_loc],dim=1)
        feat_comp_loc = self.net_shared(concat_feat)
        outputs = self.classifer(feat_comp_loc)

        return common_feat_comp,common_feat_loc,feat_comp_loc,outputs

    def finetune(self, model_path:str):
        load_model(self, model_path)

    def freeze_comp_net(self):
        for param in self.net_comp.parameters():
            param.requires_grad = False

    def freeze_loc_net(self):
        for param in self.net_loc.parameters():
            param.requires_grad = False

    def freeze_net(self):
        self.freeze_comp_net()
        self.freeze_loc_net()


class NaiveDLwEmbedding(nn.Module):
    """
    2 class classification model
    """

    def __init__(self, feat_comp_dim=102, feat_loc_dim=23, embedding_num=2405):
        super().__init__()
        self._common_feat_dim = 64
        self._embedding_dim = 64
        self._embedding_num = embedding_num
        self._feat_comp_dim = feat_comp_dim
        self._feat_loc_dim = feat_loc_dim
        self.net_comp = nn.Sequential(
            nn.Linear(feat_comp_dim, 256, bias=True),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=True),
            nn.LeakyReLU(),
            nn.Linear(128, self._common_feat_dim, bias=True),
            nn.LeakyReLU(),
        )

        self.net_emb = nn.Embedding(num_embeddings=embedding_num, embedding_dim=self._embedding_dim)

        self.net_loc_base = nn.Sequential(
            nn.Linear(feat_loc_dim, 64, bias=True),
            nn.LeakyReLU(),
        )

        self.net_loc_upper = nn.Sequential(
            nn.Linear(64+self._embedding_dim, self._common_feat_dim, bias=True),
            nn.LeakyReLU(),
        )

        self.net_shared = nn.Sequential(
            nn.Linear( self._common_feat_dim, self._common_feat_dim, bias=True),
            nn.LeakyReLU(),
        )

        self.classifer = nn.Linear(self._common_feat_dim, 2, bias=False)

    def forward(self, feat_comp, feat_loc, id_loc):
        assert (feat_comp.shape[1] == self._feat_comp_dim)
        assert (feat_loc.shape[1] == self._feat_loc_dim)
        common_feat_comp = self.net_comp(feat_comp)

        base_feat_loc = self.net_loc_upper(feat_loc)
        embed_feat_loc = self.net_emb(id_loc)

        merge_feat_loc = torch.cat([base_feat_loc,embed_feat_loc],dim=1)
        common_feat_loc = self.net_loc_upper(merge_feat_loc)

        #feature merge
        diff_feat = torch.abs(common_feat_comp - common_feat_loc)
        feat_comp_loc = self.net_shared(diff_feat)

        outputs = self.classifer(feat_comp_loc)

        return common_feat_comp, common_feat_loc, feat_comp_loc, outputs

    def finetune(self, model_path: str):
        load_model(self, model_path)

    def freeze_comp_net(self):
        for param in self.net_comp.parameters():
            param.requires_grad = False

    def freeze_loc_net(self):
        for param in self.net_loc.parameters():
            param.requires_grad = False

    def freeze_net(self):
        self.freeze_comp_net()
        self.freeze_loc_net()

#===================================================================================================
#===================================================================================================
# CNN Toy
#===================================================================================================
#===================================================================================================
class CnnToyNet(nn.Module):
    def __init__(self, num_classes, img_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(3,2,1)
        )
        self._scale = 8
        self._fc_len = img_size // self._scale * img_size // self._scale * 32

        self.fc1 = nn.Linear(self._fc_len, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input):
        x = self.net(input)
        #         print(x.shape)
        x = x.view(x.shape[0],-1)
        feat1 = self.fc1(x)
        leaky_feat1 = F.leaky_relu(feat1)
        feat2 = self.fc2(leaky_feat1)

        return feat1, feat2


# ===================================================================================================
# ===================================================================================================
# CNN Toy + vlad
# ===================================================================================================
# ===================================================================================================
class CnnVladToyNet(nn.Module):
    def __init__(self, num_classes, img_size=64):
        super().__init__()
        last_conv_ch = 32
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(),
            nn.Conv2d(64, last_conv_ch, kernel_size=1, padding=0),
            nn.LeakyReLU(),
            nn.AvgPool2d(3, 2, 1)
        )
        self._scale = 8
        # self._fc_len = img_size // self._scale * img_size // self._scale * 32

        self.vlad = NetVladLayer(input_channels=last_conv_ch,centers_num=32)
        self.fc2 = nn.Linear(self.vlad.output_features, num_classes)

    def forward(self, input):
        x = self.net(input)
        feat1 = self.vlad(x)
        feat2 = self.fc2(feat1)

        return feat1, feat2
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
        fc2 = self.dict_layer(fc1,targets)
        fc3 = self.last_layer(fc2)
        loss = self.dict_layer.getLoss()
        return fc2,fc3,loss

class ResNetV4(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False,net_cls=M.resnet50, dropout=False):
        super(ResNetV4,self).__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Linear(self.net.fc.in_features, 1024),
                nn.Dropout(),
                nn.LeakyReLU(),
            )
        else:
            self.net.fc = nn.Sequential(
                nn.Linear(self.net.fc.in_features, 1024),
                nn.LeakyReLU(),
            )

        self.fc_layer = nn.Linear(1024, 1024)
        self.last_layer = nn.Linear(1024, num_classes)

    #freeze param
    def freeze(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        fc1 = self.net(x)
        fc2 = self.fc_layer(fc1)
        fc2 = F.leaky_relu(fc2)
        fc3 = self.last_layer(fc2)
        return fc2,fc3

    def finetuning(self,num_classes):
        #only change the last layer
        self.last_layer = nn.Linear(1024,num_classes)

# model.classifier = nn.Sequential(*[model.classifier[i] for i in range(4)])
# print(model.classifier)
# model.classifier = nn.Sequential(*list(model.classifier.children())[:-3])
class ResNetVlad(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.resnet50, dropout=False, centerK = 32):
        super(ResNetVlad, self).__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.net = nn.Sequential(*list(self.net.children())[:-2]) #get rid of last 2 layer, and conv is last layer

        self.conv = nn.Conv2d(512,64,(1,1))
        self.netvlad = NetVladLayer(64,centers_num=centerK)
        self.netvladlen = self.netvlad.output_features
        self.last_layer = nn.Linear(self.netvladlen,num_classes)

    # freeze param
    def freeze(self):
        for param in self.net.parameters():
            param.requires_grad = False

    def fresh_params(self):
        return list(self.netvlad.parameters(),self.net.last_layer.parameters())

    def forward(self, x):
        fc1 = self.net(x)
        fc1 = self.conv(fc1)
        fc1 = F.leaky_relu(fc1)

        netvladfeat = self.netvlad(fc1)

        clsfeat = self.last_layer(netvladfeat)
        return netvladfeat, clsfeat

    def finetuning(self, num_classes):
        # only change the last layer
        self.last_layer = nn.Linear(self.netvladlen, num_classes)

##======================================================================================================================
##ResNet50 + BBox from Alibaba Image Search
##======================================================================================================================
class ResNetBBoxMask(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False,net_cls=M.resnet50):
        super(ResNetBBoxMask,self).__init__()
        self.netBBox = create_net(net_cls, pretrained=pretrained)
        self.netBBox.avgpool = AvgPool()
        self.netBBox.fc = nn.Linear(self.netBBox.fc.in_features, 4)

        self.mask_layer = maskLayer()

        self.netCls = create_net(net_cls, pretrained=pretrained)
        self.netCls.avgpool = AvgPool()
        self.netCls.fc = nn.Sequential(
            nn.Linear(self.netCls.fc.in_features, 1024),
            nn.LeakyReLU(),
        )
        self.fc_layer = nn.Linear(1024, 1024)
        self.last_layer = nn.Linear(1024, num_classes)

    def forward(self, x):
        bbox = self.netBBox(x)
        bbox = torch.sigmoid(bbox)

        x_hat = self.mask_layer(bbox,x.shape[3])
        x_hat = x_hat.expand(-1,3,-1,-1)

        next_x = x*x_hat

        fc1 = self.netCls(next_x)
        fc2 = self.fc_layer(fc1)
        fc2 = F.leaky_relu(fc2)
        fc3 = self.last_layer(fc2)
        return fc2,fc3,bbox,x_hat

    def fresh_params(self):
        return self.last_layer.parameters()

    def finetuning(self,num_classes):
        #only change the last layer
        self.last_layer = nn.Linear(1024,num_classes)



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

class DenseNetV2(nn.Module):
    def __init__(self, num_classes,
                 pretrained=False, net_cls=M.densenet121,dropout=False):
        super().__init__()
        self.net = create_net(net_cls, pretrained=pretrained)
        self.avg_pool = AvgPool()
        # self.net.classifier = nn.Linear(
        #     self.net.classifier.in_features, num_classes)
        if dropout:
            self.net.classifier = nn.Sequential(
                nn.Linear(self.net.classifier.in_features, 1024),
                nn.Dropout(),
                nn.LeakyReLU(),
            )
        else:
            self.net.classifier = nn.Sequential(
                nn.Linear(self.net.classifier.in_features, 1024),
                nn.LeakyReLU(),
            )

        self.fc_layer = nn.Linear(1024, 1024)
        self.last_layer = nn.Linear(1024, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        out = self.fc_layer(out)
        fc2 = F.leaky_relu(out)
        out = self.last_layer(fc2)
        return out

    def finetuning(self,num_classes):
        #only change the last layer
        self.last_layer = nn.Linear(1024,num_classes)

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
inception_v4_netvlad = partial(inceptionv4_netvlad)
inception_v4_attention = partial(inceptionv4_attention)
vggnetvlad = partial(VggNetVLAD)


inceptionresnet_v2 = partial(inceptionresnetv2)
dpn_92 = partial(dpn92)
dpn_68b = partial(dpn68b)

resnet18 = partial(ResNet, net_cls=M.resnet18)
resnetvlad18 = partial(ResNetVlad,net_cls=M.resnet18)
resnet18V4 = partial(ResNetV4, net_cls=M.resnet18)
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

cnntoynet = partial(CnnToyNet)
cnnvladtoynet = partial(CnnVladToyNet)

location_recommend_model_v1 = partial(NaiveDL)
location_recommend_model_v3 = partial(NaiveDLwEmbedding)

