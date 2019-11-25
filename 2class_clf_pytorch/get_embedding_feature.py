import os,sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import json
from pathlib import Path
import shutil
from typing import Dict
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score
import torch
from torch import nn, cuda
from torch.optim import Adam, SGD
import tqdm
import models.location_recommendation as rsmodels
from dataset import TrainDatasetLocationRS,collate_TrainDatasetLocationRS
from utils import (write_event, load_model, ThreadingDataLoader as DataLoader, adjust_learning_rate,
                   ON_KAGGLE)
from gunlib.company_location_score_lib import translocname2dict

from models.utils import *
from udf.basic import list2str
from udf.basic import save_obj,load_obj,calc_topk_acc_cat_all,topk_recall_score_all

# from torch.utils.data import DataLoader

pjoin = os.path.join
#not used @this version /home/ubuntu/location_recommender_system/ /Users/yefeichen/Database/location_recommender_system
TR_DATA_ROOT = '/Users/yefeichen/Database/location_recommender_system/'
TT_DATA_ROOT = '/home/ubuntu/location_recommender_system/'

OLD_N_CLASSES = 2
N_CLASSES = 2#253#109

nPosTr = 1000
nNegTr = 2000

model_name = '' #same as main cmd --model XXX
wework_location_only = True



#=============================================================================================================================
#main
#=============================================================================================================================
def main():
    #cmd and arg parser
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', choices=['input_grad'], default='input_grad')
    arg('--run_root', default='result/location_recommend_model_v6_5city_191113')
    arg('--model', default='location_recommend_model_v6')
    arg('--ckpt', type=str, default='model_loss_best.pt')
    arg('--finetuning',action='store_true')
    arg('--apps',type=str,default='_191113.csv')
    arg('--pre_name', type=str, default='sampled_ww_')



    #cuda version T/F
    use_cuda = cuda.is_available()

    args = parser.parse_args()
    #run_root: model/weights root
    run_root = Path(args.run_root)


    global model_name
    model_name = args.model

    df_loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT,'location_feat'+args.apps),index_col=0)

    loc_name_dict = translocname2dict(df_loc_feat)
    print('Location Embedding Number: %d'%len(loc_name_dict))

    # se- ception dpn can only use finetuned model from imagenet
    # model = getattr(models, args.model)(feat_comp_dim=102, feat_loc_dim=23) #location_recommend_model_v1
    model = getattr(rsmodels, args.model)(feat_comp_dim=102,feat_loc_dim=23,embedding_num=len(loc_name_dict)) #location_recommend_model_v3

    md_path = Path(str(run_root) + '/' + args.ckpt)
    if md_path.exists():
        print('load weights from md_path')
        load_model(model, md_path)

    model.freeze()

    dict_len = len(loc_name_dict)

    loc_id = torch.LongTensor(np.array(range(dict_len))).reshape(-1,1)

    if use_cuda:
        loc_id = loc_id.cuda()

    emb_vecs = model.net_emb(loc_id).reshape(dict_len,-1)
    # deep_feat = self.net_deep(embed_feat)

    emb_vecs = emb_vecs.data.cpu().numpy()

    loc_num,feat_dim = emb_vecs.shape
    print('loc num:%d,feature dims:%d'%(loc_num,feat_dim))

    feat_cols = [ 'feat#'+str(c) for c in range(feat_dim) ]
    feat_dat = pd.DataFrame(emb_vecs,columns=feat_cols)

    loc_name = [ c for c in loc_name_dict.keys() ]
    name_dat = pd.DataFrame(loc_name,columns=['atlas_location_uuid'])

    loc_dat = pd.concat([name_dat,feat_dat],axis=1)
    loc_dat.to_csv('location_feat_emb.csv')


#=============================================================================================================================
#End of main
#=============================================================================================================================

# #=============================================================================================================================
# #predict
# #=============================================================================================================================
def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]


def softmax_loss(results, labels):
    labels = labels.view(-1)
    loss = F.cross_entropy(results, labels, reduce=True)

    return loss


def softmax_lossV2(results,labels):

    softmax_label = labels[labels < N_CLASSES].view(-1)
    label_len = softmax_label.shape[0]
    softmax_results = results[:label_len,:]
    assert(label_len%2==0)
    loss = F.cross_entropy(softmax_results,softmax_label,reduce=True)

    return loss




if __name__ == '__main__':
    main()


