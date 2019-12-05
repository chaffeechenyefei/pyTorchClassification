import os,sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
from itertools import islice
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
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from utils import (write_event, load_model, ThreadingDataLoader as DataLoader, adjust_learning_rate,
                   ON_KAGGLE)
from gunlib.company_location_score_lib import translocname2dict

from models.utils import *
from udf.basic import list2str
from udf.basic import save_obj,load_obj,calc_topk_acc_cat_all,topk_recall_score_all
import matplotlib.pyplot as plt
from gunlib.company_location_score_lib import global_filter,sub_rec_similar_company,sub_rec_condition,merge_rec_reason_rowise,reason_json_format

# from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

pjoin = os.path.join
#not used @this version /home/ubuntu/location_recommender_system/ /Users/yefeichen/Database/location_recommender_system
TR_DATA_ROOT = '/home/ubuntu/location_recommender_system/'
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
    arg('--fold', type=int, default=0)
    arg('--model', default='location_recommend_model_v6')
    arg('--ckpt', type=str, default='model_loss_best.pt')
    arg('--pretrained', type=str, default='imagenet')#resnet 1, resnext imagenet
    arg('--batch-size', type=int, default=1)
    arg('--step', type=str, default=8)#update the gradients every 8 batch(sample num = step*batch-size*inner_size)
    arg('--workers', type=int, default=16)
    arg('--lr', type=float, default=3e-4)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=1)
    arg('--epoch-size', type=int)
    arg('--tta', type=int, default=1)
    arg('--use-sample', action='store_true', help='use a sample of the dataset')
    arg('--debug', action='store_true')
    arg('--imgsize',type=int, default = 256)
    arg('--finetuning',action='store_true')
    arg('--cos_sim_loss',action='store_true')
    arg('--ensemble', action='store_true')
    arg('--sample_rate',type=float,default=1.0)#sample part of testing data for evaluating during training
    arg('--testStep',type=int,default=500000)
    arg('--query_location',action='store_true',help='use location as query')
    arg('--apps',type=str,default='_191113.csv')
    arg('--pre_name', type=str, default='sampled_ww_')



    #cuda version T/F
    use_cuda = cuda.is_available()

    args = parser.parse_args()
    #run_root: model/weights root
    run_root = Path(args.run_root)


    global model_name
    model_name = args.model

    df_comp_feat = pd.read_csv(pjoin(TR_DATA_ROOT,'company_feat'+args.apps),index_col=0)
    df_loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT,'location_feat'+args.apps),index_col=0)

    feat_name = list(df_comp_feat.columns)[1:]+list(df_loc_feat.columns)[1:]
    print(len(feat_name))

    clfile = ['PA', 'SF', 'SJ', 'LA', 'NY']
    cfile = ['dnb_pa.csv', 'dnb_sf.csv', 'dnb_sj.csv', 'dnb_Los_Angeles.csv', 'dnb_New_York.csv']
    lfile = 'location_scorecard_191113.csv'

    clfile = [c + args.apps for c in clfile]
    pre_name = args.pre_name
    pred_save_name = [ pre_name + c.replace(args.apps,'') + '_similarity'+args.apps for c in clfile ]

    #Dont use ensemble score
    df_ensemble = pd.DataFrame(columns=['Blank'])

    loc_name_dict = translocname2dict(df_loc_feat)
    print('Location Embedding Number: %d'%len(loc_name_dict))

    ##::DataLoader
    def make_loader(df_comp_feat: pd.DataFrame, df_loc_feat: pd.DataFrame, df_pair: pd.DataFrame, emb_dict:dict,df_ensemble,
                    name='train',flag_ensemble=args.ensemble,testStep=args.testStep,shuffle=True) -> DataLoader:
        return DataLoader(
            TrainDatasetLocationRS(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_pair,df_ensemble_score=df_ensemble,
                                   emb_dict=emb_dict, name=name,flag_ensemble=flag_ensemble,
                                   negN=nNegTr, posN=nPosTr, testStep=testStep),
            shuffle=shuffle,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_TrainDatasetLocationRS
        )

    #Not used in this version
    criterion = softmax_loss
    lossType = 'softmax'

    # se- ception dpn can only use finetuned model from imagenet
    # model = getattr(models, args.model)(feat_comp_dim=102, feat_loc_dim=23) #location_recommend_model_v1
    model = getattr(rsmodels, args.model)(feat_comp_dim=102,feat_loc_dim=23,embedding_num=len(loc_name_dict)) #location_recommend_model_v3

    md_path = Path(str(run_root) + '/' + args.ckpt)
    if md_path.exists():
        print('load weights from md_path')
        load_model(model, md_path)

    model.freeze()

    all_params = filter(lambda p: p.requires_grad, model.parameters())


    #gpu first
    if use_cuda:
        model = model.cuda()

    #print(model)
    if args.mode == 'input_grad':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        Path(str(run_root) + '/params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        for ind_city in range(5):
            print('Operating %s...'%pred_save_name[ind_city])
            testing_pair = pd.read_csv(pjoin(TR_DATA_ROOT, pred_save_name[ind_city]))[['atlas_location_uuid', 'duns_number']]
            testing_pair['label'] = 0
            testing_pair = testing_pair[['duns_number', 'atlas_location_uuid','label']]

            predict_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=testing_pair,
                                   emb_dict=loc_name_dict,df_ensemble=df_ensemble, name='valid',shuffle=False)

            predict_with_reason(model=model,criterion=criterion,predict_loader=predict_loader,use_cuda=use_cuda,test_pair=testing_pair,
                                feat_name=feat_name,save_name=pred_save_name[ind_city],pre_name='dlsub_')


#=============================================================================================================================
#End of main
#=============================================================================================================================

# #=============================================================================================================================
# #predict
# #=============================================================================================================================
def merge_col_ind(x,col_name,feat_name,topk=3):
    merged_cols = []
    cnt = 0
    for i in col_name:
        im_feat = feat_name[int(x[i])]
        if cnt >= topk:
            break
        else:
            if not (im_feat.startswith('primary') or im_feat.startswith('major_')):
                merged_cols.append(feat_name[int(x[i])])
                cnt +=1
    return list2str(merged_cols)

def predict_with_reason(
        model: nn.Module, criterion, predict_loader, use_cuda, test_pair, feat_name, save_name: str, pre_name: str = ''):
    model.train()
    topk_feature = len(feat_name)
    col_name = []
    col_name = ['reason_topk' + str(i) for i in range(topk_feature)]
    all_losses, all_predictions, all_targets, all_comp_feats, all_loc_feats,all_x_grads= [], [], [],[],[],[]
    # with torch.no_grad():
    for batch_dat in predict_loader:
        featComp = batch_dat['feat_comp']
        featLoc = batch_dat['feat_loc']
        featId = batch_dat['feat_id']
        targets = batch_dat['target']
        featEnsemble = batch_dat['feat_ensemble_score']
        all_targets.append(targets)  # torch@cpu
        if use_cuda:
            featComp, featLoc, targets, featId, featEnsemble = featComp.cuda(), featLoc.cuda(), targets.cuda(), featId.cuda(), featEnsemble.cuda()
        #d_loss/d_inputs
        featComp.requires_grad,featLoc.requires_grad = True,True

        model_output = model(feat_comp=featComp, feat_loc=featLoc, id_loc=featId, feat_ensemble_score=featEnsemble)
        outputs = model_output['outputs']

        targets = torch.zeros_like(targets,dtype=targets.dtype)

        loss = softmax_loss(outputs, targets)

        batch_size = featComp.size(0)
        (batch_size * loss).backward()

        x_grad = torch.cat([featComp.grad,featLoc.grad],dim=1)
        all_x_grads.append( x_grad.abs() )

        all_losses.append(loss.data.cpu().numpy())


    all_x_grads = torch.cat(all_x_grads)
    print('all_predictions.shape: ',all_x_grads.shape)

    all_x_grads = all_x_grads.data.cpu().numpy()
    #get topk index of column
    all_x_grads = all_x_grads.argsort(axis=1)[:,:-topk_feature-1:-1]

    assert(all_x_grads.shape[1]==len(col_name))


    print('saving...')
    dat_grad_pd = pd.DataFrame(data=all_x_grads, columns=col_name)
    res_pd = pd.concat([test_pair, dat_grad_pd], axis=1)

    res_pd['merged_feat'] = res_pd.apply(lambda x:merge_col_ind(x,col_name,feat_name),axis=1)
    res_pd = res_pd[['duns_number', 'atlas_location_uuid','merged_feat']]

    res_pd.to_csv(pjoin(TR_DATA_ROOT,pre_name+save_name))

    return res_pd



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


