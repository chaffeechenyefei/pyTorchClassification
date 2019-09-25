import torch
import torch.nn as nn
import torch.nn.functional as F
from models.netvlad import NetVladLayerV2
from utils import load_model_with_dict


model_path = '/home/ubuntu/pytorch/2class_clf_pytorch/result/checkpoint.pth.tar'

tNet = NetVladLayerV2(num_clusters= 64, dim = 512)
load_model_with_dict(tNet, model_path, 'pool.', '')

input = torch.rand(512,512,1,1)

with torch.no_grad():
    k1 = tNet(input)
    k2 = tNet(input)


print((k1-k2).sum())









