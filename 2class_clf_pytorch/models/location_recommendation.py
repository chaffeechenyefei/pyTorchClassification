from functools import partial
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
import math
import torch.utils.model_zoo as model_zoo
from utils import load_model_with_dict_replace, load_model_with_dict, load_model
from collections import OrderedDict
from models.utils import idx_2_one_hot
from torch import cuda

use_cuda = cuda.is_available()


# ===================================================================================================
# ===================================================================================================
# Basic module for any number of input channel with L size
# ===================================================================================================
# ===================================================================================================
"""
setLinearLayerConv1d
setLinearLayer
setLinearLayer_fast
Those 3 are different implementation of region model.
"""
class setLinearLayerConv1d(nn.Module):
    """
    mapping a set of feature into one unified vector vesion 1
    input: [B,K,Lin]
    output:[B,Lout]
    conv1d version
    """

    def __init__(self, fin: int, fout: int):
        super().__init__()
        self._fin = fin
        self._fout = fout
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=fout, kernel_size=fin),
            nn.LeakyReLU(),
        )

    def forward(self, feat_set, type='mean'):
        """
        here we consider the combination of feature as mean-pooling
        """
        B, K, fin = feat_set.shape
        assert (fin == self._fin)
        if type == 'mean':
            m_feat_set = feat_set.sum(dim=1, keepdim=True) / K  # [B,K,fin]->[B,1,fin]
            unified_feat = self.net(m_feat_set)  # [B,1,fin]->[B,fout,1]
            return unified_feat.squeeze()
        elif type == 'mean_debug':
            feat_out = 0
            for k in range(K):
                feat_out += self.net(feat_set[:, k, :].unsqueeze(1))  # [B,1,fin]->[B,fout,1]
            feat_out /= K
            return feat_out.squeeze()
        else:  # max pooling
            feat_out = torch.zeros(B, self._fout, K)  # [B,fout,K]
            if use_cuda:
                feat_out = feat_out.cuda()
            for k in range(K):
                feat_out[:, :, k] = 1.0 * self.net(feat_set[:, k, :].unsqueeze(1)).squeeze()  # [B,1,fin]->[B,fout,1]
            pool = F.max_pool1d
            feat_out = pool(feat_out, K)
            return feat_out.squeeze()


class setLinearLayer(nn.Module):
    """
    mapping a set of feature into one unified vector vesion 2
    input: [B,K,Lin]
    output:[B,Lout,Lmid] here Lout is consider as channel and Lmid is real feature
    this version will use linear layer
    """

    def __init__(self, fin: int, fout: int, fmid: int = 1):
        super().__init__()
        self._fin = fin
        self._fout = fout
        self._fmid = fmid
        self.nets = []
        for i in range(self._fout):
            self.nets.append(nn.Linear(in_features=fin, out_features=self._fmid))
            # [B,*,Lin]->[B,*,Lout] weights are shared for each *

    def forward(self, feat_set):
        B, K, fin = feat_set.shape
        assert (fin == self._fin)
        feat_out = torch.zeros(B, self._fout, self._fmid)  # [B,fout,fmid]
        pool = F.max_pool1d
        for i in range(self._fout):
            feat_mid = self.nets[i](feat_set)  # [B,K,fin]->[B,K,fmid]
            feat_mid = pool(feat_mid.permute(0, 2, 1),
                            K).squeeze()  # [B,K,fmid]-permute>[B,fmid,K]-pool>[B,fmid,1]->[B,fmid]
            feat_out[:, i, :] = feat_mid
        # feat_out = [B,fout,fmid]
        return feat_out


class setLinearLayer_fast(nn.Module):
    """
    mapping a set of feature into one unified vector vesion 2
    input: [B,K,Lin]
    output:[B,Lout] here Lout is consider as channel and Lmid is real feature
    this version will use linear layer
    """

    def __init__(self, fin: int, fout: int,type='maxpool'):
        super().__init__()
        self._fin = fin
        self._fout = fout
        self._fmid = 1  # default 1 only supported value
        self._type = type
        self.net = nn.Sequential(
            nn.Linear(in_features=fin, out_features=fout),
            nn.LeakyReLU(),
        )

    def forward(self, feat_set):
        B, K, fin = feat_set.shape
        assert (fin == self._fin)
        feat_out = self.net(feat_set)  # [B,K,fin] -> [B,K,fout]

        if self._type in ['maxpool','max_pooling','max_pool']:
            pool = F.max_pool1d
            feat_out = pool(feat_out.permute(0, 2, 1),
                            K).squeeze()  # [B,K,fout]-permute>[B,fout,K]-pool>[B,fout,1]->[B,fout]
        else:
            feat_out = feat_out.sum(dim=1)
        return feat_out #[B,fout]


class companyMLP(nn.Module):
    """
    Network for company feature extracting.
    Just a simple MLP model. But it works for [B,*,L] shaped input
    Ref: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    """

    def __init__(self, fid: int, fod: int, l2norm: bool = True):
        super().__init__()
        self._fid = fid
        self._fod = fod
        self.Flagl2norm = l2norm
        self.mlp = nn.Sequential(
            nn.Linear(in_features=fid, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=fod),
            nn.LeakyReLU(),
        )

    def forward(self, feat_set):
        """
        :param feat_set: [B,K,fid] 
        where:
            B:  bath_size 
            K:  number of points/companies
            fid: dimensionality of input feature
        :return: 
        """
        if self.Flagl2norm:
            if len(feat_set.shape) == 3:
                dim = 2
            elif len(feat_set.shape) == 2:
                dim = 1
            else:
                assert (0)
            feat_set = F.normalize(feat_set, p=2, dim=dim)

        return self.mlp(feat_set)  # [B,K,fod]


class RegionModelv1(nn.Module):
    """
    
    """

    def __init__(self, feat_comp_dim=102):
        super().__init__()
        self.emb_feat_comp_dim = 64
        self.feat_region_dim = 32
        self._feat_comp_dim = feat_comp_dim
        self.netEmb = companyMLP(fid=feat_comp_dim, fod=self.emb_feat_comp_dim)
        self.netReg = setLinearLayerConv1d(fin=self.emb_feat_comp_dim, fout=self.feat_region_dim)
        self.netSng = nn.Sequential(
            nn.Linear(in_features=self.emb_feat_comp_dim, out_features=self.feat_region_dim),
            nn.LeakyReLU()
        )
        feat_cls_dim = 32
        self.netClf = nn.Sequential(
            nn.Linear(in_features=self.feat_region_dim + self.feat_region_dim, out_features=feat_cls_dim),
            nn.LeakyReLU(),
            nn.Linear(in_features=feat_cls_dim, out_features=2)
        )

    def forward(self, feat_comp, feat_K_comp):
        emb_feat_comp = self.netEmb(feat_comp)
        emb_feat_K_comp = self.netEmb(feat_K_comp)

        single_feat_comp = self.netSng(emb_feat_comp)
        region_feat_comp = self.netReg(feat_set=emb_feat_K_comp, type='maxpooling')

        concat_feat = torch.cat([single_feat_comp, region_feat_comp], dim=1)
        outputs = self.netClf(concat_feat)

        return {
            'feat_comp': emb_feat_comp,
            'feat_region': region_feat_comp,
            'outputs': outputs
        }


class RegionModelv2(nn.Module):
    """

    """

    def __init__(self, feat_comp_dim=102, feat_loc_dim=23):
        super().__init__()
        self.emb_feat_comp_dim = 64
        self.feat_region_dim = 32
        self.emb_feat_loc_dim = 32
        self._feat_comp_dim = feat_comp_dim
        self._feat_loc_dim = feat_loc_dim

        self.netEmb = companyMLP(fid=feat_comp_dim, fod=self.emb_feat_comp_dim)

        self.netLoc = nn.Sequential(
            nn.Linear(feat_loc_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, self.emb_feat_loc_dim),
            nn.LeakyReLU(),
        )

        self.netReg = setLinearLayerConv1d(fin=self.emb_feat_comp_dim, fout=self.feat_region_dim)

        self.netSng = nn.Sequential(
            nn.Linear(in_features=self.emb_feat_comp_dim, out_features=self.feat_region_dim),
            nn.LeakyReLU()
        )

        feat_cls_dim = 32

        self.netClf = nn.Sequential(
            nn.Linear(in_features=self.feat_region_dim + self.feat_region_dim + self.emb_feat_loc_dim,
                      out_features=feat_cls_dim),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=feat_cls_dim, out_features=2)
        )

    def forward(self, feat_comp, feat_K_comp, feat_loc):
        emb_feat_comp = self.netEmb(feat_comp)
        emb_feat_K_comp = self.netEmb(feat_K_comp)

        single_feat_comp = self.netSng(emb_feat_comp)
        region_feat_comp = self.netReg(feat_set=emb_feat_K_comp, type='maxpooling')

        emb_feat_loc = self.netLoc(feat_loc)

        concat_feat = torch.cat([single_feat_comp, region_feat_comp, emb_feat_loc], dim=1)
        outputs = self.netClf(concat_feat)

        return {
            'feat_comp': emb_feat_comp,
            'feat_region': region_feat_comp,
            'outputs': outputs
        }


class RegionModelv3(nn.Module):
    """
    location id embedding is replaced by region model
    others remain same as Deep and Wide
    """

    def __init__(self, feat_comp_dim=102, feat_loc_dim=23):
        super().__init__()
        self.emb_feat_comp_dim = 64
        self.feat_region_dim = 64

        self.netEmb = companyMLP(fid=feat_comp_dim, fod=self.emb_feat_comp_dim)
        self.netReg = setLinearLayerConv1d(fin=self.emb_feat_comp_dim, fout=self.feat_region_dim)
        self.netDeep = nn.Sequential(
            nn.Linear(in_features=self.feat_region_dim, out_features=self.feat_region_dim),
            nn.LeakyReLU()
        )

        self.netClf = nn.Sequential(
            nn.Linear(in_features=self.feat_region_dim + feat_comp_dim + feat_loc_dim, out_features=64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, feat_comp, feat_K_comp, feat_loc):
        emb_feat_K_comp = self.netEmb(feat_K_comp)
        region_feat_comp_org = self.netReg(feat_set=emb_feat_K_comp, type='maxpooling')
        region_feat_comp = self.netDeep(region_feat_comp_org)
        feat_deep_and_wide = torch.cat([feat_comp, feat_loc, region_feat_comp], dim=1)
        outputs = self.netClf(feat_deep_and_wide)
        return {
            'feat_region_org': region_feat_comp_org,
            'feat_region_deep': region_feat_comp,
            'outputs': outputs
        }


class RegionModelv4(nn.Module):
    """
    location id embedding is replaced by region model
    others remain same as Deep and Wide
    additional loss based on the idea that embedded vector should contain infor abt the location itself
    """

    def __init__(self, feat_comp_dim=102, feat_loc_dim=23):
        super().__init__()
        self.emb_feat_comp_dim = 64
        self.feat_region_dim = 64

        self.netEmb = companyMLP(fid=feat_comp_dim, fod=self.emb_feat_comp_dim)
        self.netReg = setLinearLayerConv1d(fin=self.emb_feat_comp_dim, fout=self.feat_region_dim)
        self.netDeep = nn.Sequential(
            nn.Linear(in_features=self.feat_region_dim, out_features=self.feat_region_dim),
            nn.LeakyReLU()
        )
        self.netDecoder = nn.Sequential(
            nn.Linear(in_features=self.feat_region_dim, out_features=feat_loc_dim),
        )

        self.netClf = nn.Sequential(
            nn.Linear(in_features=self.feat_region_dim + feat_comp_dim + feat_loc_dim, out_features=64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, feat_comp, feat_K_comp, feat_loc):
        emb_feat_K_comp = self.netEmb(feat_K_comp)
        region_feat_comp_org = self.netReg(feat_set=emb_feat_K_comp, type='maxpooling')
        explicit_feat_loc = self.netDecoder(region_feat_comp_org)
        region_feat_comp = self.netDeep(region_feat_comp_org)
        if feat_comp is not None and feat_loc is not None:
            feat_deep_and_wide = torch.cat([feat_comp, feat_loc, region_feat_comp], dim=1)
            outputs = self.netClf(feat_deep_and_wide)
        else:
            outputs = torch.zeros(feat_K_comp.shape[0], 2)
            if use_cuda:
                outputs = outputs.cuda()
        return {
            'feat_region_org': region_feat_comp_org,
            'feat_region_deep': region_feat_comp,
            'feat_loc_pred': explicit_feat_loc,
            'outputs': outputs
        }

class RegionModelv5(nn.Module):
    """
    location id embedding is replaced by region model
    others remain same as Deep and Wide
    additional loss based on the idea that embedded vector should contain infor abt the location itself
    region model part is accelerated by using setLinearLayer_fast
    fast version of RegionModelv4
    """

    def __init__(self, feat_comp_dim=102, feat_loc_dim=23):
        super().__init__()
        self.emb_feat_comp_dim = 64
        self.feat_region_dim = 64

        self.netEmb = companyMLP(fid=feat_comp_dim, fod=self.emb_feat_comp_dim)
        self.netReg = setLinearLayer_fast(fin=self.emb_feat_comp_dim, fout=self.feat_region_dim,type='maxpool')
        self.netDeep = nn.Sequential(
            nn.Linear(in_features=self.feat_region_dim, out_features=self.feat_region_dim),
            nn.LeakyReLU()
        )
        self.netDecoder = nn.Sequential(
            nn.Linear(in_features=self.feat_region_dim, out_features=feat_loc_dim),
        )

        self.netClf = nn.Sequential(
            nn.Linear(in_features=self.feat_region_dim + feat_comp_dim + feat_loc_dim, out_features=64),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self, feat_comp, feat_K_comp, feat_loc):
        emb_feat_K_comp = self.netEmb(feat_K_comp)
        region_feat_comp_org = self.netReg(feat_set=emb_feat_K_comp)
        explicit_feat_loc = self.netDecoder(region_feat_comp_org)
        region_feat_comp = self.netDeep(region_feat_comp_org)
        if feat_comp is not None and feat_loc is not None:
            feat_deep_and_wide = torch.cat([feat_comp, feat_loc, region_feat_comp], dim=1)
            outputs = self.netClf(feat_deep_and_wide)
        else:
            outputs = torch.zeros(feat_K_comp.shape[0], 2)
            if use_cuda:
                outputs = outputs.cuda()
        return {
            'feat_region_org': region_feat_comp_org,
            'feat_region_deep': region_feat_comp,
            'feat_loc_pred': explicit_feat_loc,
            'outputs': outputs
        }

    def predict(self,feat_comp,feat_region,feat_loc):
        region_feat_comp = self.netDeep(feat_region)
        feat_deep_and_wide = torch.cat([feat_comp, feat_loc, region_feat_comp], dim=1)
        outputs = self.netClf(feat_deep_and_wide)

        return {
            'feat_region_deep':region_feat_comp,
            'outputs':outputs
        }


class NaiveLR(nn.Module):
    """
    no embedding, no region modeling
    """

    def __init__(self, feat_comp_dim=102, feat_loc_dim=23):
        super().__init__()
        self.netClf = nn.Sequential(
            nn.Linear(in_features=feat_comp_dim + feat_loc_dim, out_features=2),
        )

    def forward(self, feat_comp, feat_loc):
        feat_wide = torch.cat([feat_comp, feat_loc], dim=1)
        outputs = self.netClf(feat_wide)

        return {
            'outputs': outputs
        }


# ===================================================================================================
# ===================================================================================================
# Location Recommendation Model
# ===================================================================================================
# ===================================================================================================
class NaiveDL(nn.Module):
    """
    2 class classification model
    """

    def __init__(self, feat_comp_dim=102, feat_loc_dim=23):
        super().__init__()
        self._common_feat_dim = 64
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

        self.net_loc = nn.Sequential(
            nn.Linear(feat_loc_dim, 64, bias=True),
            nn.LeakyReLU(),
            nn.Linear(64, self._common_feat_dim, bias=True),
            nn.LeakyReLU(),
        )

        self.net_shared = nn.Sequential(
            nn.Linear(2 * self._common_feat_dim, self._common_feat_dim, bias=True),
            nn.LeakyReLU(),
        )

        self.classifer = nn.Linear(self._common_feat_dim, 2, bias=False)

    def forward(self, feat_comp, feat_loc):
        assert (feat_comp.shape[1] == self._feat_comp_dim)
        assert (feat_loc.shape[1] == self._feat_loc_dim)
        common_feat_comp = self.net_comp(feat_comp)
        common_feat_loc = self.net_loc(feat_loc)
        concat_feat = torch.cat([common_feat_comp, common_feat_loc], dim=1)
        feat_comp_loc = self.net_shared(concat_feat)
        outputs = self.classifer(feat_comp_loc)
        return {
            'comp_feat': common_feat_comp,
            'loc_feat': common_feat_loc,
            'comp_loc_feat': feat_comp_loc,
            'outputs': outputs
        }

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
            nn.Linear(64 + self._embedding_dim, self._common_feat_dim, bias=True),
            nn.LeakyReLU(),
        )

        self.net_shared = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self._common_feat_dim, self._common_feat_dim, bias=True),
            nn.LeakyReLU(),
        )

        self.classifer = nn.Linear(self._common_feat_dim, 2, bias=False)

    def forward(self, feat_comp, feat_loc, id_loc):
        assert (feat_comp.shape[1] == self._feat_comp_dim)
        assert (feat_loc.shape[1] == self._feat_loc_dim)
        common_feat_comp = self.net_comp(feat_comp)

        base_feat_loc = self.net_loc_base(feat_loc)
        embed_feat_loc = self.net_emb(id_loc).view(-1, self._embedding_dim)

        merge_feat_loc = torch.cat([base_feat_loc, embed_feat_loc], dim=1)
        common_feat_loc = self.net_loc_upper(merge_feat_loc)

        # feature merge
        diff_feat = torch.abs(common_feat_comp - common_feat_loc)
        feat_comp_loc = self.net_shared(diff_feat)

        outputs = self.classifer(feat_comp_loc)

        return {
            'comp_feat': common_feat_comp,
            'loc_feat': common_feat_loc,
            'comp_loc_feat': feat_comp_loc,
            'outputs': outputs
        }

    def finetune(self, model_path: str):
        load_model(self, model_path)

    def freeze_comp_net(self):
        for param in self.net_comp.parameters():
            param.requires_grad = False

    def freeze_loc_net(self):
        for param in self.net_loc_base.parameters():
            param.requires_grad = False
        for param in self.net_loc_upper.parameters():
            param.requires_grad = False

    def freeze_net(self):
        self.freeze_comp_net()
        self.freeze_loc_net()


class NaiveDLwEmbedding_concat(nn.Module):
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
            nn.Linear(64 + self._embedding_dim, self._common_feat_dim, bias=True),
            nn.LeakyReLU(),
        )

        self.net_shared = nn.Sequential(
            nn.Linear(2 * self._common_feat_dim, self._common_feat_dim, bias=True),
            nn.LeakyReLU(),
        )

        self.classifer = nn.Linear(self._common_feat_dim, 2, bias=False)

    def forward(self, feat_comp, feat_loc, id_loc):
        assert (feat_comp.shape[1] == self._feat_comp_dim)
        assert (feat_loc.shape[1] == self._feat_loc_dim)
        common_feat_comp = self.net_comp(feat_comp)

        base_feat_loc = self.net_loc_base(feat_loc)
        embed_feat_loc = self.net_emb(id_loc).view(-1, self._embedding_dim)

        merge_feat_loc = torch.cat([base_feat_loc, embed_feat_loc], dim=1)
        common_feat_loc = self.net_loc_upper(merge_feat_loc)

        # feature merge
        concat_feat = torch.cat([common_feat_comp, common_feat_loc], dim=1)
        feat_comp_loc = self.net_shared(concat_feat)

        outputs = self.classifer(feat_comp_loc)

        return {
            'comp_feat': common_feat_comp,
            'loc_feat': common_feat_loc,
            'comp_loc_feat': feat_comp_loc,
            'outputs': outputs
        }

    def finetune(self, model_path: str):
        load_model(self, model_path)

    def freeze_comp_net(self):
        for param in self.net_comp.parameters():
            param.requires_grad = False

    def freeze_loc_net(self):
        for param in self.net_loc_base.parameters():
            param.requires_grad = False
        for param in self.net_loc_upper.parameters():
            param.requires_grad = False

    def freeze_net(self):
        self.freeze_comp_net()
        self.freeze_loc_net()


class NaiveDeepWide(nn.Module):
    """
    2 class classification model for deep and wide
    """

    def __init__(self, feat_comp_dim=102, feat_loc_dim=23, embedding_num=2405, feat_ensemble_dim=0):
        super().__init__()
        self._embedding_dim = 64
        self._embedding_num = embedding_num
        self._feat_comp_dim = feat_comp_dim
        self._feat_loc_dim = feat_loc_dim
        self._feat_ensemble_dim = feat_ensemble_dim

        self._deep_feat_dim = 64
        self._wide_feat_dim = self._feat_comp_dim + self._feat_loc_dim
        self.net_emb = nn.Embedding(num_embeddings=embedding_num, embedding_dim=self._embedding_dim)
        self.net_deep = nn.Sequential(
            nn.Linear(self._embedding_dim, self._deep_feat_dim),
            nn.LeakyReLU()
        )

        self.net_shared = nn.Sequential(
            nn.Linear(self._wide_feat_dim + self._deep_feat_dim + self._feat_ensemble_dim, 64),
            nn.Dropout(p=0.1),
            nn.LeakyReLU(),
        )

        self.classifer = nn.Sequential(
            nn.Linear(64, 2),
        )

    def forward(self, feat_comp, feat_loc, id_loc, feat_ensemble_score=None):
        wide_feat = torch.cat([feat_comp, feat_loc], dim=1)
        embed_feat = self.net_emb(id_loc).view(-1, self._embedding_dim)
        deep_feat = self.net_deep(embed_feat)
        if feat_ensemble_score is not None and self._feat_ensemble_dim == feat_ensemble_score.shape[1]:
            all_feat = torch.cat([wide_feat, deep_feat, feat_ensemble_score], dim=1)
        else:
            all_feat = torch.cat([wide_feat, deep_feat], dim=1)
        all_feat = self.net_shared(all_feat)
        outputs = self.classifer(all_feat)

        return {
            'outputs': outputs,
            'wide_feat': wide_feat,
            'deep_feat': deep_feat
        }

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class NaiveDLCosineLosswKemb(nn.Module):
    """
    2 class classification model
    """

    def __init__(self, feat_comp_dim=102, feat_loc_dim=23, embedding_num=2405):
        super().__init__()
        self._common_feat_dim = 96
        self._embedding_dict_dim = 80
        self._embedding_dict_num = 10
        self._embedding_dim = self._embedding_dict_dim * self._embedding_dict_num

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

        self.net_loc = nn.Sequential(
            nn.Linear(feat_loc_dim, 16, bias=True),
            nn.LeakyReLU(),
        )

        self.classifer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(self._common_feat_dim, 2, bias=False)
        )

    def forward(self, feat_comp, feat_loc, id_loc, feat_ensemble_score=None):
        assert (feat_comp.shape[1] == self._feat_comp_dim)
        assert (feat_loc.shape[1] == self._feat_loc_dim)

        v_comp = self.net_comp(feat_comp)  # [B,96]
        v_comp = v_comp.unsqueeze(dim=1)  # [B,1,96]
        v_emb = self.net_emb(id_loc)  # [B,80*10]
        v_loc = self.net_loc(feat_loc)  # [B,16]
        v_loc = v_loc.unsqueeze(dim=1)  # [B,1,16]

        v_loc_b = v_loc.expand(-1, self._embedding_dict_num, -1)  # [B,K,16]
        v_emb = v_emb.view(-1, self._embedding_dict_num, self._embedding_dict_dim)  # [B,K,80]

        v_loc_concat = torch.cat([v_loc_b, v_emb], dim=2)  # [B,K,96]

        v_cos_dist = F.cosine_similarity(v_comp, v_loc_concat, dim=2)  # [B,K]
        outputs_cos, max_id = torch.max(v_cos_dist, dim=1, keepdim=True)  # [B,1],[B,1]

        dum_max_id = idx_2_one_hot(max_id, self._embedding_dict_num, use_cuda=use_cuda)  # [B,K]
        dum_max_id = dum_max_id.unsqueeze(dim=2).expand(-1, -1, self._common_feat_dim)  # [B,K,96]
        dum_max_id = dum_max_id > 0
        v_loc_concat_max = v_loc_concat[dum_max_id].reshape(-1, self._common_feat_dim)  # [B,96]

        v_diff = torch.abs(v_loc_concat_max - v_comp.squeeze())

        outputs_cls = self.classifer(v_diff)

        return {
            'outputs': outputs_cls,
            'outputs_cos': outputs_cos,
            'comp_feat': v_comp.squeeze(),
            'loc_set_feat': v_loc_concat,
            'loc_feat': v_loc_concat_max,
        }

    def finetune(self, model_path: str):
        load_model(self, model_path)


location_recommend_model_v1 = partial(NaiveDL)
location_recommend_model_v2 = partial(NaiveDLwEmbedding_concat)
location_recommend_model_v3 = partial(NaiveDLwEmbedding)
location_recommend_model_v4 = partial(NaiveDeepWide)
location_recommend_model_v5 = partial(NaiveDLCosineLosswKemb)
location_recommend_model_v6 = partial(NaiveDeepWide)  # They use similar structure

location_recommend_region_model_v1 = partial(RegionModelv1)
location_recommend_region_model_v2 = partial(RegionModelv2)
location_recommend_region_model_v3 = partial(RegionModelv3)
location_recommend_region_model_v4 = partial(RegionModelv4)
location_recommend_region_model_v5 = partial(RegionModelv5)
location_recommend_region_model_v0 = partial(NaiveLR)
