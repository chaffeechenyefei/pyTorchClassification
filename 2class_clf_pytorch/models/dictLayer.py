import torch
import torch.nn as nn
from models.utils import *

FLAG_CUDA = torch.cuda.is_available()

def repCol(matA,n:int):
    matB = []
    for i in range(0,matA.size(1)):
        for j in range(0,n):
            matB.append(matA[:,i])
    matC  = torch.stack(matB,dim=1)
    return matC




class DictLayer(nn.Module):
    def __init__(self,input_features,output_features,nCls,alpha = 0.05,bias=True):
        super(DictLayer,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.nCls = nCls
        self._alpha = alpha
        self._dictloss = 0

        self.weights = nn.Parameter(torch.Tensor(output_features,input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        self.weights.data.uniform_(-0.1,0.1)
        if bias is not None:
            self.bias.data.uniform_(-0.1,0.1)


    def forward(self,inputs,targets):
        N = inputs.size(0)
        eachGroup = self.output_features // self.nCls
        #inputs:[N,input_features] x [output_features, input_features]^T
        inputs = inputs.view(N,-1)
        y = torch.mm(inputs,self.weights.t()) #y:[N,output_features]

        if self.bias is not None:
            y = y + self.bias.view(1,-1).expand(N,self.output_features)

        #suppose targets are discrete
        h = idx_2_one_hot(targets,self.nCls, use_cuda = FLAG_CUDA)#h:[N,nCls]

        h_mat = repCol(h,eachGroup)#[N,output_features]

        h_res = 1.0 - h_mat

        dictLoss = y.mul(h_res)
        dictLoss = self._alpha*torch.norm(dictLoss)

        self._dictloss = dictLoss/N

        return y

    def getLoss(self):
        return self._dictloss


class DictConv2dLayer(nn.Module):
    def __init__(self,in_channels,out_channels,nCls,kernel_size,stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros',alpha = 0.05):
        super(DictConv2dLayer,self).__init__()
        self. _in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups
        self._bias = bias
        self._padding_mode = padding_mode
        self._nCls = nCls
        self._alpha = alpha
        self._dictloss = 0.0

        self.conv2d_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias , padding_mode)

    def forward(self,inputs,targets):
        N = inputs.size(0)
        C = inputs.size(1)
        eachGroup = self._out_channels // self._nCls
        targets = targets.view(-1,1)

        y = self.conv2d_layer(inputs)#[N,C,H,W]
        #suppose targets are discrete
        # h = idx_2_one_hot(targets,self.nCls, use_cuda = True)#h:[N,nCls]

        h_mat = torch.ones_like(y) #[N,C, H,W]
        for b in range(0,N):
            t = int(targets[b,0].item())
            h_mat[b,t,:,:] = 0.0

        dictLoss = y.mul(h_mat)
        dictLoss = self._alpha*torch.norm(dictLoss)

        self._dictloss = dictLoss/N

        return y

    def getLoss(self):
        return self._dictloss




        


