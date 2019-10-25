import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from models.lovasz import *

import numpy as np

##========================================================================
##Basic function
##========================================================================
def idx_2_one_hot(y_idx,nCls,use_cuda=False):
    """
    y_idx:  LongTensor shape:[1,batch_size] or [batch_size,1]
    y_one_hot:FloatTensor shape:[batch_size, nCls]
    """
    y_idx = y_idx.long().view(-1,1)
    batch_size = y_idx.shape[0]
    y_one_hot = torch.zeros(batch_size,nCls)
    if use_cuda:
        y_one_hot = y_one_hot.cuda()
    y_one_hot.scatter_(1, y_idx, 1.)
    return y_one_hot

def one_hot_2_idx(y_one_hot):
    """
    y_one_hot:FloatTensor shape:[batch_size, nCls]
    y_idx: LongTensor shape:[batch_size,1]
    """
    _,idx_mat = torch.max(y_one_hot,1,keepdim=False)
    y_idx = idx_mat.view(-1,1)
    return y_idx

def one_hot_2_idx_numpy(y_one_hot):
    """
    y_one_hot:numpy shape:[batch_size, nCls]
    y_idx: numpy shape:[batch_size,1]
    """
    idx_mat = np.argmax(y_one_hot,1)
    y_idx = idx_mat.reshape(-1,1)
    return y_idx

def topkAcc(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k
    output: [N,nCls]
    target: [1,N]or[N,1] discrete
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)#pred = [N,maxk]
    pred = pred.t()#pred = [maxk,N]
    correct = pred.eq(target.view(1, -1).expand_as(pred))#target.view(1, -1).expand_as(pred): [1,N]=>[maxk,N]

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
##========================================================================
##Basic Loss Module
##========================================================================
class TripletLossV1(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False,discreteTarget=True):
        super(TripletLossV1, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin,reduction='mean')
        self.mutual = mutual_flag
        self.targetsFlag = discreteTarget #if True then labels with shape (1,batch_size). if False then labels with shape (batch_size,class_num)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size) //( each value is class idx)
        """
        n = inputs.size(0)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())#1*dist-2*inputs@inputs.t
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        if self.targetsFlag:
            targets = targets.view(1,-1)#(1,batch_size)
            mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        else:#one hot ----> discreate
            _,idx_mat = torch.max(targets,1,keepdim=False)
            mask = idx_mat.expand(n,n).eq(idx_mat.expand(n,n).t())

        # For each anchor, find the hardest positive and negative
        
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss


class TripletLossV2(nn.Module):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3, mutual_flag=False):
        super(TripletLossV2, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin, reduction='mean')
        self.mutual = mutual_flag


    def forward(self, inputs, targets):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (batch_size) //( each value is class idx)
        """
        # _targets = targets[targets>=N_CLASSES].view(1,-1)
        # Len = _targets.shape[1]
        #
        # _inputs = inputs[-Len:,:]


        n = inputs.size(0)
        targets = targets.view(1,-1)
        # inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())  # 1*dist-2*inputs@inputs.t
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())


        # For each anchor, find the hardest positive and negative

        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        if self.mutual:
            return loss, dist
        return loss

# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2.0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         if isinstance(alpha, (float, int)):
#             self.alpha = torch.Tensor([alpha, 1 - alpha])
#         if isinstance(alpha, list):
#             self.alpha = torch.Tensor(alpha)
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim() > 2:
#             # N,C,H,W => N,C,H*W
#             input = input.view(input.size(0), input.size(1), -1)
#             input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1, 1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1, target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         if self.alpha is not None:
#             if self.alpha.type() != input.data.type():
#                 self.alpha = self.alpha.type_as(input.data)
#             at = self.alpha.gather(0, target.data.view(-1))
#             logpt = logpt * Variable(at)

#         loss = -1 * (1 - pt)**self.gamma * logpt
#         if self.size_average:
#             return loss.mean()
#         else:
#             return loss.sum()


class FocalLoss_BCE(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True,discreteTarget=False,nCls=-1):
        super(FocalLoss_BCE, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.targetsFlag = discreteTarget
        self._nCls = nCls


    def forward(self, input, target):#target with shape (N, cls_num)
        if input.dim() > 2:
            # N,C,H,W => N,C,H*W
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))   # N,H*W,C => N*H*W,C
        if targetsFlag:#discrete-->contineous
            y = idx_2_one_hot(target, self._nCls, use_cuda = True)
        else:
            y = target.view(-1)#<-> pt.view(-1) and BCE is point-wise

        # pt = torch.sigmoid(input)
        pt = input
        pt = pt.view(-1)
        error = torch.abs(pt - y)
        log_error = torch.log(error)
        loss = -1 * (1 - error)**self.gamma * log_error
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def try_bestfitting_loss(results, labels, selected_num=10):
    batch_size, class_num = results.shape
    labels = labels.view(-1, 1)
    one_hot_target = torch.zeros(
        batch_size, class_num + 1).cuda().scatter_(1, labels, 1)[:, :5004].contiguous()
    error_loss = lovasz_hinge(results, one_hot_target)
    labels = labels.view(-1)
    indexs_new = (labels != 5004).nonzero().view(-1)
    if len(indexs_new) == 0:
        return error_loss
    results_nonew = results[torch.arange(0, len(results))[
        indexs_new], labels[indexs_new]].contiguous()
    target_nonew = torch.ones_like(results_nonew).float().cuda()
    nonew_loss = nn.BCEWithLogitsLoss(reduce=True)(results_nonew, target_nonew)
    return nonew_loss + error_loss


def sigmoid_loss(results, labels, topk=10):
    if len(results.shape) == 1:
        results = results.view(1, -1)
    batch_size, class_num = results.shape
    labels = labels.view(-1, 1)
    one_hot_target = torch.zeros(
        batch_size, class_num + 1).cuda().scatter_(1, labels, 1)[:, :5004 * 2]
    #lovasz_loss = lovasz_hinge(results, one_hot_target)
    error = torch.abs(one_hot_target - torch.sigmoid(results))
    error = error.topk(topk, 1, True, True)[0].contiguous()
    target_error = torch.zeros_like(error).float().cuda()
    error_loss = nn.BCELoss(reduce=True)(error, target_error)
    labels = labels.view(-1)
    indexs_new = (labels != 5004 * 2).nonzero().view(-1)
    if len(indexs_new) == 0:
        return error_loss
    results_nonew = results[torch.arange(0, len(results))[
        indexs_new], labels[indexs_new]].contiguous()
    target_nonew = torch.ones_like(results_nonew).float().cuda()
    nonew_loss = nn.BCEWithLogitsLoss(reduce=True)(results_nonew, target_nonew)
    return nonew_loss + error_loss


if __name__ == '__main__':
    results = torch.randn((4, 5004)).cuda()
    targets = torch.from_numpy(np.array([1, 2, 3, 5004])).cuda()
    print(try_bestfitting_loss(results, targets))
