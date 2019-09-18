from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from models.netvlad import NetVladLayer
from models.maskLayer import *
from utils import *
import os
import sys

__all__ = ['InceptionV4', 'inceptionv4', 'inceptionv4_netvlad','inceptionv4_attention']

pretrained_settings = {
    'inceptionv4': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(64, 96, kernel_size=3, stride=2)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 64, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(64, 64, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(64, 96, kernel_size=(3,3), stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(192, 192, kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(384, 96, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(384, 64, kernel_size=1, stride=1),
            BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(384, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(384, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=3, stride=1, padding=1),
            BasicConv2d(224, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(1024, 384, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 256, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(192, 224, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(224, 224, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(224, 256, kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1024, 128, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(1024, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(1024, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(256, 320, kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(320, 320, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(1536, 256, kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(384, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(384, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch2_0 = BasicConv2d(1536, 384, kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(384, 448, kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2 = BasicConv2d(448, 512, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(512, 256, kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(512, 256, kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(1536, 256, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionV4(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(), # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )
        self.last_linear = nn.Linear(1536, num_classes,bias=False)
        self.feature_map = None
        self._infer_mode = False

    def logits(self, features):
        #Allows image of any size to be processed
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        feat_triplet = x.view(x.size(0), -1)
        x = self.last_linear(feat_triplet)
        return x,feat_triplet

    def get_featuremap(self):
        if self.feature_map is not None:
            self.feature_map = torch.sum(self.feature_map, dim=1)

    def set_infer_mode(self):
        self._infer_mode = True

    def forward(self, input):
        x = self.features(input)

        if self._infer_mode:
            x_dim = x.shape[1]
            msk = torch.sum(x,dim=1,keepdim=True) #[B,1,H,W]
            # msk = 1000*( ( msk - msk.min() ) / ( msk.max() - msk.min() ) - 0.1 )
            msk = (( msk - msk.min() ) / ( msk.max() - msk.min() ) > 0.01).float()
            msk = msk.expand((-1,x_dim,-1,-1)) #[B,...,H,W]
            x = msk*x

        self.feature_map = x
        x,feat_triplet = self.logits(x)
        return feat_triplet,x

    def changelastlayer(self, num_classes):
        self.last_linear = nn.Linear(1536,num_classes,bias=False)

    def finetune(self,model_path:str):
        load_model_with_dict_replace(self,model_path,'basemodel.','')

    def freeze_net(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def initial(self):
        settings = pretrained_settings['inceptionv4']['imagenet']
        pretrained_dict = model_zoo.load_url(settings['url'])

        cls_dict = self.state_dict()
        pretrained_cls_dict = {k: v for k, v in pretrained_dict.items() if k in cls_dict and 'last_linear' not in k}
        cls_dict.update(pretrained_cls_dict)
        self.load_state_dict(cls_dict)

def inceptionv4(num_classes=1000, pretrained='imagenet'):
    if pretrained:
        settings = pretrained_settings['inceptionv4'][pretrained]
#         assert num_classes == settings['num_classes'], \
#             "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionV4(num_classes=num_classes)
        
        pretrained_dict = model_zoo.load_url(settings['url'])
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'last_linear' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, num_classes,bias=False)
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionV4(num_classes=num_classes)
    return model

#=======================================================================================================================
# InceptionV4_Attention 20190910
#=======================================================================================================================
class InceptionV4_Attention(nn.Module):
    def __init__(self, num_classes=1001):
        super(InceptionV4_Attention, self).__init__()
        self.att_net = InceptionV4(4)
        self.cls_net = InceptionV4(num_classes)
        self.mask_net = gaussianMaskLayer()
        self.sig = nn.Sigmoid()
        self.bbox = None
        self.xywh = None

    def forward(self,input):
        _,xywh = self.att_net(input)
        self.xywh = 1.0*xywh
        xywh = self.sig(xywh)
        mask = self.mask_net(xywh,imgSize=input.shape[2])
        self.bbox = 1.0*mask
        mask = mask.expand(-1, 3, -1, -1)
        mask_input = input*mask

        feat_triplet, result = self.cls_net(mask_input)

        return feat_triplet,result

    def finetune(self,model_path:str):
        load_model_with_dict_replace(self.cls_net,model_path,'basemodel.','')
        load_model_with_dict_replace(self.att_net,model_path,'basemodel.','',lastLayer=False)

    def freeze_clsnet(self):
        for param in self.cls_net.parameters():
            param.requires_grad = False

    def freeze_attnet_radius(self):
        self.mask_net.freeze_radius()

    def initial(self):
        settings = pretrained_settings['inceptionv4']['imagenet']
        pretrained_dict = model_zoo.load_url(settings['url'])

        cls_dict = self.cls_net.state_dict()
        pretrained_cls_dict = {k: v for k, v in pretrained_dict.items() if k in cls_dict and 'last_linear' not in k}
        cls_dict.update(pretrained_cls_dict)
        self.cls_net.load_state_dict(cls_dict)

        att_dict = self.att_net.state_dict()
        pretrained_att_dict= {k: v for k, v in pretrained_dict.items() if k in att_dict and 'last_linear' not in k}
        att_dict.update(pretrained_att_dict)
        self.att_net.load_state_dict(att_dict)


def inceptionv4_attention(num_classes=1000, pretrained='imagenet'):
    model = InceptionV4_Attention(num_classes)
    return model
#=======================================================================================================================
# InceptionV4_Attention Ends
#=======================================================================================================================

#=======================================================================================================================
# InceptionV4_NetVlad 20190909
#=======================================================================================================================
class InceptionV4_NetVlad(nn.Module):

    def __init__(self, num_classes=1001):
        super(InceptionV4_NetVlad, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        self.dictnum = 128
        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3, stride=1),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            Mixed_3a(),
            Mixed_4a(),
            Mixed_5a(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Inception_A(),
            Reduction_A(), # Mixed_6a
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Inception_B(),
            Reduction_B(), # Mixed_7a
            Inception_C(),
            Inception_C(),
            Inception_C()
        )

        self.convPool = nn.Conv2d(1536,256,(1,1))
        self.netvlad = NetVladLayer(256,self.dictnum)
        self.last_linear = nn.Linear(256*self.dictnum,num_classes)


    def getvladfeat(self,features):
        x = self.convPool(features)
        x = F.leaky_relu(x)
        feat_triplet = self.netvlad(x)
        x = self.last_linear(feat_triplet)
        return x,feat_triplet

    def forward(self, input):
        x = self.features(input)
        x,feat_triplet = self.getvladfeat(x)
        return feat_triplet,x

    def changelastlayer(self, num_classes):
        self.last_linear = nn.Linear(256*self.dictnum,num_classes)

    def finetune(self,model_path:str):
        load_model_with_dict_replace(self,model_path,'basemodel.','',False)

    def freeze_net(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def initial(self):
        settings = pretrained_settings['inceptionv4']['imagenet']
        pretrained_dict = model_zoo.load_url(settings['url'])

        cls_dict = self.state_dict()
        pretrained_cls_dict = {k: v for k, v in pretrained_dict.items() if k in cls_dict and 'last_linear' not in k}
        cls_dict.update(pretrained_cls_dict)
        self.load_state_dict(cls_dict)


def inceptionv4_netvlad(num_classes=1000, pretrained='imagenet'):
    model = InceptionV4_NetVlad(num_classes=num_classes)
    return model
#=======================================================================================================================
# InceptionV4_NetVlad Ends
#=======================================================================================================================
'''
TEST
Run this code with:
```
cd $HOME/pretrained-models.pytorch
python -m pretrainedmodels.inceptionv4
```
'''
if __name__ == '__main__':

    assert inceptionv4(num_classes=10, pretrained=None)
    print('success')
    assert inceptionv4(num_classes=1000, pretrained='imagenet')
    print('success')
    assert inceptionv4(num_classes=1001, pretrained='imagenet+background')
    print('success')

    # fail
    assert inceptionv4(num_classes=1001, pretrained='imagenet')