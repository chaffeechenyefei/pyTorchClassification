import torch
import torch.nn as nn
from models.utils import *

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
		y = torch.mm(inputs,self.weights.t()) #y:[N,output_features]

		if self.bias is not None:
			y = y + self.bias.view(1,-1).expand(N,self.output_features)

		#suppose targets are discrete
		h = idx_2_one_hot(targets,nCls)#h:[N,nCls]

		h_mat = repCol(h,eachGroup)#[N,output_features]

		h_res = 1.0 - h_mat

		dictLoss = y.mul(h_res)
		dictLoss = self._alpha*torch.norm(dictLoss)

		self._dictloss = dictLoss/N

		return y

		def getLoss(self):
			return self._dictloss



