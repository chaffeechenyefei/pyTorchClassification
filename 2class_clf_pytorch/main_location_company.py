import os,sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import warnings
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
from udf.basic import save_obj,load_obj,calc_topk_acc_cat_all,topk_recall_score_all
import matplotlib.pyplot as plt
from gunlib.company_location_score_lib import global_filter,sub_rec_similar_company,sub_rec_condition,merge_rec_reason_rowise,reason_json_format

# from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

pjoin = os.path.join
#not used @this version
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
    arg('--mode', choices=['train', 'validate', 'predict_valid', 'predict_test','train_test', 'predict_sub','predict_salesforce'], default='train')
    arg('--run_root', default='result/location_company')
    arg('--fold', type=int, default=0)
    arg('--model', default='location_recommend_model_v3')
    arg('--ckpt', type=str, default='model_loss_best.pt')
    arg('--pretrained', type=str, default='imagenet')#resnet 1, resnext imagenet
    arg('--batch-size', type=int, default=1)
    arg('--step', type=str, default=8)#update the gradients every 8 batch(sample num = step*batch-size*inner_size)
    arg('--workers', type=int, default=16)
    arg('--lr', type=float, default=3e-4)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=80)
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
    arg('--apps',type=str,default='_191114.csv')



    #cuda version T/F
    use_cuda = cuda.is_available()

    args = parser.parse_args()
    #run_root: model/weights root
    run_root = Path(args.run_root)


    global model_name
    model_name = args.model

    df_all_pair = pd.read_csv(pjoin(TR_DATA_ROOT,'train_val_test_location_company_82split'+args.apps),index_col=0)
    df_comp_feat = pd.read_csv(pjoin(TR_DATA_ROOT,'company_feat'+args.apps),index_col=0)
    df_loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT,'location_feat'+args.apps),index_col=0)

    clfile = ['PA', 'SF', 'SJ', 'LA', 'NY']
    cityname = ['Palo Alto', 'San Francisco', 'San Jose', 'Los Angeles', 'New York']
    c_salesforce_file = 'salesforce_comp_city_from_opp.csv'
    cfile = ['dnb_pa.csv', 'dnb_sf.csv', 'dnb_sj.csv', 'dnb_Los_Angeles.csv', 'dnb_New_York.csv']
    lfile = 'location_scorecard_191113.csv'

    clfile = [c + args.apps for c in clfile]
    pred_save_name = [ c.replace(args.apps,'') + '_similarity'+args.apps for c in clfile ]

    if args.ensemble:
        df_ensemble = []
        ensemble_score = ['PA_comp_loc_score_exclude_self.csv','SJ_comp_loc_score_exclude_self.csv',
                          'SF_comp_loc_score_exclude_self.csv',
                          'LA_comp_loc_score_exclude_self.csv','NY_comp_loc_score_exclude_self.csv']
        for i in range(len(ensemble_score)):
            df_ensemble.append(pd.read_csv(pjoin(TR_DATA_ROOT,ensemble_score[i])))
        df_ensemble = pd.concat(df_ensemble,axis=0).reset_index(drop=True)
    else:
        df_ensemble = pd.DataFrame(columns=['Blank'])

    #col error 'unnamed:0'

    #Not used @this version...
    train_root = TR_DATA_ROOT
    valid_root = TT_DATA_ROOT

    #split train/valid fold
    df_train_pair = df_all_pair[df_all_pair['fold'] == 0]
    if args.mode == 'train':
        df_valid_pair = df_all_pair[df_all_pair['fold'] == 2].sample(frac=args.sample_rate).reset_index(drop=True)
        print('Validate size%d'%len(df_valid_pair))
    else:
        df_valid_pair = df_all_pair[df_all_pair['fold'] == 2]
        print('Validate size%d' % len(df_valid_pair))
        if args.query_location:
            df_valid_pair = df_valid_pair.sort_values(by=['atlas_location_uuid']).reset_index(drop=True)
    del df_all_pair

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
    # criterion = nn.BCEWithLogitsLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss(reduction='none')
    if args.cos_sim_loss:
        criterion = nn.CosineEmbeddingLoss(0.2,reduce=True,reduction='mean')
        lossType = 'cosine'
    else:
        criterion = softmax_loss
        lossType = 'softmax'

    # se- ception dpn can only use finetuned model from imagenet
    # model = getattr(models, args.model)(feat_comp_dim=102, feat_loc_dim=23) #location_recommend_model_v1
    model = getattr(rsmodels, args.model)(feat_comp_dim=102,feat_loc_dim=23,embedding_num=len(loc_name_dict)) #location_recommend_model_v3

    md_path = Path(str(run_root) + '/' + args.ckpt)
    if md_path.exists():
        print('load weights from md_path')
        load_model(model, md_path)

    ##params::Add here
    #params list[models.parameters()]
    # all_params = list(model.parameters())
    all_params = filter(lambda p: p.requires_grad, model.parameters())

    #apply parallel gpu if available
    # model = torch.nn.DataParallel(model)

    #gpu first
    if use_cuda:
        model = model.cuda()

    #print(model)
    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        Path(str(run_root) + '/params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_train_pair, emb_dict=loc_name_dict, df_ensemble=df_ensemble, name='train')
        valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_valid_pair, emb_dict=loc_name_dict, df_ensemble=df_ensemble, name='valid')

        train_kwargs = dict(
            args=args,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            patience=args.patience,
            init_optimizer=lambda params, lr: Adam(params, lr, betas=(0.9,0.999), eps=1e-08, weight_decay = 2e-4),
            use_cuda=use_cuda,
        )

        train(params=all_params, **train_kwargs)

    elif args.mode == 'train_test':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        Path(str(run_root) + '/params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_train_pair, emb_dict=loc_name_dict, df_ensemble=df_ensemble, name='train')
        valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_valid_pair, emb_dict=loc_name_dict, df_ensemble=df_ensemble, name='valid')

        train_kwargs = dict(
            args=args,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            patience=args.patience,
            init_optimizer=lambda params, lr: Adam(params, lr, betas=(0.9,0.999), eps=1e-08, weight_decay = 2e-4),
            use_cuda=use_cuda,
        )

        train(params=all_params, **train_kwargs)

        for ind_city in [3,4]:
            pdcl = pd.read_csv(pjoin(TR_DATA_ROOT, clfile[ind_city]))[['atlas_location_uuid', 'duns_number']]
            # pdcl = pdcl.groupby(['atlas_location_uuid', 'duns_number']).first().reset_index(drop=True)
            # pdcl = pdcl.groupby('atlas_location_uuid').first().reset_index()
            pdc = pd.read_csv(pjoin(TR_DATA_ROOT, cfile[ind_city]))[['duns_number']]
            pdc['atlas_location_uuid'] = 'a'
            all_loc_name = pdcl[['atlas_location_uuid']].groupby(['atlas_location_uuid'])[
                ['atlas_location_uuid']].first().reset_index(drop=True)
            all_loc_name['key'] = 0
            pdc['key'] = 0

            testing_pair = pd.merge(pdc, all_loc_name, on='key', how='left',
                                    suffixes=['_left', '_right']).reset_index(drop=True)

            testing_pair = testing_pair.rename(
                columns={'atlas_location_uuid_left': 'groundtruth', 'atlas_location_uuid_right': 'atlas_location_uuid'})
            testing_pair = testing_pair[['duns_number', 'atlas_location_uuid','groundtruth']]
            testing_pair['label'] = (testing_pair['atlas_location_uuid'] == testing_pair['groundtruth'])
            testing_pair = testing_pair[['duns_number', 'atlas_location_uuid','label']]

            valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=testing_pair,
                                   emb_dict=loc_name_dict,df_ensemble=df_ensemble, name='valid',shuffle=False)
            print('Predictions for city %d' % ind_city)
            predict(model,criterion,tqdm.tqdm(valid_loader, desc='Validation'),
                    use_cuda=use_cuda,test_pair=testing_pair[['atlas_location_uuid', 'duns_number']], save_name= pred_save_name[ind_city] , lossType=lossType)

    elif args.mode == 'predict_sub':
        """
        """
        for ind_city in range(5):
            print('Processing City:%s'%cityname[ind_city])
            comp_loc = pd.read_csv(pjoin(TR_DATA_ROOT, clfile[ind_city]))[['atlas_location_uuid', 'duns_number']]
            comp_feat = pd.read_csv(pjoin(TR_DATA_ROOT,cfile[ind_city]))
            loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT,lfile))

            #we only focus on buildings with companies inside
            loc_feat = loc_feat.merge(comp_loc[['atlas_location_uuid']].groupby('atlas_location_uuid').first().reset_index(), on='atlas_location_uuid',how='inner',suffixes=['','_right'])

            print('Filtering')
            #Global filter: master degree!
            global_ft = global_filter(loc_feat=loc_feat)
            sub_loc_feat = global_ft\
                .city_filter(city_name=cityname[ind_city]).end()
                # .filtering(key_column='pct_masters_degree',percentile=0.5)\
                # .filtering('score_accessibility',0.6).end()


            if wework_location_only:
                sub_loc_feat_ww = sub_loc_feat[sub_loc_feat['is_wework']==True]
                sub_comp_loc = pd.merge(comp_loc,sub_loc_feat_ww[['atlas_location_uuid']],on='atlas_location_uuid',how='inner',suffixes=['','_right'])
            else:
                sub_comp_loc = pd.merge(comp_loc, sub_loc_feat[['atlas_location_uuid']], on='atlas_location_uuid',
                                        how='inner', suffixes=['', '_right'])

            print('remaining companies:%d'%len(sub_comp_loc))

            print('Reasoning')
            #Reason 1:
            print('Customized Reason')
            matching_col = 'primary_sic_2_digit'
            recall_com1 = sub_rec_similar_company(comp_feat=comp_feat, comp_loc=sub_comp_loc,
                                                  matching_col=matching_col,reason_col_name='reason')
            sub_pairs = recall_com1.get_candidate_location_for_company(query_comp_feat=comp_feat)
            print('sub_pairs:%d'%len(sub_pairs))

            #Reason2:
            print('General Reason')
            recall_com2 = sub_rec_condition(sub_loc_feat)
            sub_loc_recall_com2 = recall_com2.exfiltering('num_fitness_gyms', percentile=0.5, reason='Enough GYM')
            sub_loc_recall_com3 = recall_com2.exfiltering('num_drinking_places', percentile=0.5,
                                                          reason='Entertainment Available')
            sub_loc_recall_com4 = recall_com2.exfiltering('num_eating_places', percentile=0.5, reason='Easy for lunch')
            print('recall_location_size: %d, %d, %d'%(len(sub_loc_recall_com2), len(sub_loc_recall_com3), len(sub_loc_recall_com4.shape)))

            print('Merging general reasons')
            sub_loc_recall = pd.concat([sub_loc_recall_com2, sub_loc_recall_com3, sub_loc_recall_com4], axis=0)
            sub_loc_recall = merge_rec_reason_rowise(sub_loc_recall, group_cols=['atlas_location_uuid'],
                                                     merge_col='reason')
            if wework_location_only:
                sub_loc_recall = sub_loc_recall.merge(sub_loc_feat_ww[['atlas_location_uuid']],on='atlas_location_uuid',how='inner',suffixes=['','_right'])
            print('sub_loc_recall sized %d' % len(sub_loc_recall))
            # print(sub_loc_recall.columns)

            print('Making pairs for general reasons')
            comp_feat['key'] = 0
            sub_loc_recall['key'] = 0
            sub_pairs2 = pd.merge(comp_feat[['duns_number', 'key']], sub_loc_recall, on='key', how='left',
                                       suffixes=['', '_right'])

            ##merging reason TODO
            print('Union Customized Reason and general reason')
            sub_pairs = pd.concat([sub_pairs, sub_pairs2], axis=0, sort=False)
            ##not very efficient
            sub_pairs = merge_rec_reason_rowise(sub_pairs, group_cols=['duns_number', 'atlas_location_uuid'],
                                                    merge_col='reason')
            sub_pairs['label'] = 0
            print('Total sub_pairs:%d' % len(sub_pairs))

            valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=sub_pairs,
                                   emb_dict=loc_name_dict,df_ensemble=df_ensemble, name='valid',shuffle=False)
            print('Predictions for city %d' % ind_city)
            predict(model,criterion,tqdm.tqdm(valid_loader, desc='Validation'),
                    use_cuda=use_cuda,test_pair=sub_pairs[['atlas_location_uuid', 'duns_number','reason']], \
                    save_name= pred_save_name[ind_city] ,pre_name='sub_',sampling=False ,lossType=lossType)

    elif args.mode == 'validate':
        """
        For test set that each company in test set x all locations only.
        Because topks recall can not be calculated.
        But if set topks = [0,0,0], roc_auc can be calculated.
        """
        topks = [300,1000,600]

        Query_Company = not args.query_location
        for ind_city in range(3):
            df_valid_pair_city = df_valid_pair[df_valid_pair['city']==ind_city]
            all_company_loc = pd.read_csv(pjoin(TR_DATA_ROOT, clfile[ind_city]))[['atlas_location_uuid', 'duns_number']]
            loc_name = all_company_loc[['atlas_location_uuid']].groupby('atlas_location_uuid')[
                ['atlas_location_uuid']].first().reset_index(drop=True)
            num_loc = len(loc_name)
            valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_valid_pair_city,
                                   emb_dict=loc_name_dict,df_ensemble=df_ensemble, name='valid',shuffle=False)
            print('Predictions for city %d'%ind_city)
            validation(model, criterion, tqdm.tqdm(valid_loader, desc='Validation'),
                       use_cuda=use_cuda, lossType=lossType, num_loc=num_loc, topK=topks[ind_city],Query_Company=Query_Company)

    elif args.mode == 'predict_test':
        """
        It will generate a score for each company with all the locations(including companies/locations in training set)
        or locations of ww only
        """
        for ind_city in [0,1,2,3,4]:
            pdcl = pd.read_csv(pjoin(TR_DATA_ROOT, clfile[ind_city]))[['atlas_location_uuid', 'duns_number']]
            pdc = pd.read_csv(pjoin(TR_DATA_ROOT,cfile[ind_city]))[['duns_number']]
            pdc['atlas_location_uuid'] = 'a'
            # in case of multi-mapping
            # pdcl = pdcl.groupby('atlas_location_uuid').first().reset_index()
            all_loc_name = pdcl[['atlas_location_uuid']].groupby(['atlas_location_uuid'])[
                ['atlas_location_uuid']].first().reset_index(drop=True)

            if wework_location_only:
                loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT, lfile))[['atlas_location_uuid','is_wework']]
                loc_ww = loc_feat[loc_feat['is_wework']==True]
                all_loc_name = all_loc_name.merge(loc_ww,on='atlas_location_uuid',how='inner',suffixes=['','_right'])[['atlas_location_uuid']]

            all_loc_name['key'] = 0
            pdc['key'] = 0


            testing_pair = pd.merge(pdc, all_loc_name, on='key', how='left',
                                    suffixes=['_left', '_right']).reset_index(drop=True)

            testing_pair = testing_pair.rename(
                columns={'atlas_location_uuid_left': 'groundtruth', 'atlas_location_uuid_right': 'atlas_location_uuid'})
            testing_pair = testing_pair[['duns_number', 'atlas_location_uuid','groundtruth']]
            testing_pair['label'] = (testing_pair['atlas_location_uuid'] == testing_pair['groundtruth'])
            testing_pair = testing_pair[['duns_number', 'atlas_location_uuid','label']]

            valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=testing_pair,
                                   emb_dict=loc_name_dict,df_ensemble=df_ensemble, name='valid',shuffle=False)
            print('Predictions for city %d' % ind_city)

            if wework_location_only:
                pre_name = 'ww_'
            else:
                pre_name = ''

            predict(model,criterion,tqdm.tqdm(valid_loader, desc='Validation'),
                    use_cuda=use_cuda,test_pair=testing_pair[['atlas_location_uuid', 'duns_number']], pre_name=pre_name,\
                    save_name= pred_save_name[ind_city] , lossType=lossType)

    elif args.mode == 'predict_salesforce':
        """
        It will generate a score for each company with all the locations(including companies/locations in training set)
        or locations of ww only
        """
        pdc_all = pd.read_csv(pjoin(TR_DATA_ROOT, c_salesforce_file))[['duns_number','city']]
        for ind_city,str_city in enumerate(cityname):
            pdcl = pd.read_csv(pjoin(TR_DATA_ROOT, clfile[ind_city]))[['atlas_location_uuid', 'duns_number']]
            pdc = pdc_all[pdc_all['city']==str_city]
            tot_comp = len(pdc)
            print('Total %d company found in %s from salesforce'%(tot_comp,str_city))
            pdc['atlas_location_uuid'] = 'a'
            # in case of multi-mapping
            # pdcl = pdcl.groupby('atlas_location_uuid').first().reset_index()
            all_loc_name = pdcl[['atlas_location_uuid']].groupby(['atlas_location_uuid'])[
                ['atlas_location_uuid']].first().reset_index(drop=True)

            if wework_location_only:
                loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT, lfile))[['atlas_location_uuid', 'is_wework']]
                loc_ww = loc_feat[loc_feat['is_wework'] == True]
                all_loc_name = \
                all_loc_name.merge(loc_ww, on='atlas_location_uuid', how='inner', suffixes=['', '_right'])[
                    ['atlas_location_uuid']]

            tot_loc = len(all_loc_name)
            print('Total %d locations belonged to ww in %s'%(tot_loc,str_city))

            all_loc_name['key'] = 0
            pdc['key'] = 0

            testing_pair = pd.merge(pdc, all_loc_name, on='key', how='left',
                                    suffixes=['_left', '_right']).reset_index(drop=True)

            testing_pair = testing_pair.rename(
                columns={'atlas_location_uuid_left': 'groundtruth', 'atlas_location_uuid_right': 'atlas_location_uuid'})
            testing_pair = testing_pair[['duns_number', 'atlas_location_uuid', 'groundtruth']]
            testing_pair['label'] = (testing_pair['atlas_location_uuid'] == testing_pair['groundtruth'])
            testing_pair = testing_pair[['duns_number', 'atlas_location_uuid', 'label']]

            valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=testing_pair,
                                       emb_dict=loc_name_dict, df_ensemble=df_ensemble, name='valid', shuffle=False)
            print('Predictions for city %d' % ind_city)

            if wework_location_only:
                pre_name = 'ww_'
            else:
                pre_name = ''



            if args.query_location:
                topk = min(300,tot_comp)
            else:
                topk = min(3,tot_loc)

            predict(model, criterion, tqdm.tqdm(valid_loader, desc='Validation'),
                    use_cuda=use_cuda, test_pair=testing_pair[['atlas_location_uuid', 'duns_number']],
                    pre_name=pre_name, \
                    save_name=pred_save_name[ind_city], lossType=lossType,query_loc_flag=args.query_location,topk=topk)


#=============================================================================================================================
#End of main
#=============================================================================================================================

# #=============================================================================================================================
# #predict
# #=============================================================================================================================
def predict(
        model: nn.Module, criterion, predict_loader, use_cuda, test_pair, save_name:str , pre_name:str = '',topk=300,query_loc_flag=True,lossType='softmax',sampling=True) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for batch_dat in predict_loader:
            featComp = batch_dat['feat_comp']
            featLoc = batch_dat['feat_loc']
            featId = batch_dat['feat_id']
            targets = batch_dat['target']
            featEnsemble = batch_dat['feat_ensemble_score']
            all_targets.append(targets)#torch@cpu
            if use_cuda:
                featComp, featLoc, targets, featId,featEnsemble = featComp.cuda(), featLoc.cuda(), targets.cuda(), featId.cuda(),featEnsemble.cuda()
            model_output = model(feat_comp=featComp, feat_loc=featLoc, id_loc = featId,feat_ensemble_score=featEnsemble)
            outputs = model_output['outputs']

            if lossType=='softmax':
                loss = softmax_loss(outputs, targets)
                all_predictions.append(outputs)
            else:
                out_comp_feat = model_output['comp_feat']
                out_loc_feat = model_output['loc_feat']
                cos_targets = 2 * targets.float() - 1.0
                loss = criterion(out_comp_feat, out_loc_feat, cos_targets)
                all_predictions.append(model_output['outputs_cos'])

            all_losses.append(loss.data.cpu().numpy())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)#list->torch
    print('all_predictions.shape: ')
    print(all_predictions.shape)


    if lossType=='softmax':
        all_predictions = F.softmax(all_predictions,dim=1)
        all_predictions2 = all_predictions[:, 1].data.cpu().numpy()
    else:
        all_predictions = (all_predictions + 1)/2 #squeeze to [0,1]
        all_predictions2 = all_predictions.data.cpu().numpy()

    print('saving...')
    dat_pred_pd = pd.DataFrame(data=all_predictions2.reshape(-1,1), columns=['similarity'])
    res_pd = pd.concat([test_pair, dat_pred_pd], axis=1)

    if 'reason' in res_pd.columns:
        res_pd = reason_json_format(res_pd,col_name='reason')
        res_pd.rename(columns={'duns_number':'compony_id',\
                               'reason':'notes'})
        res_pd['algorithm'] = model_name

    if sampling:
        print('sampling...')
        #for each location we return topk companies
        # sample_pd = res_pd.sort_values(by=['atlas_location_uuid','similarity'],ascending=False).groupby('atlas_location_uuid').head(topk).reset_index(drop=True)
        if query_loc_flag:
            sample_pd = res_pd.groupby('atlas_location_uuid').apply(lambda x: x.nlargest(topk,['similarity'])).reset_index(drop=True)
        else:
            sample_pd = res_pd.groupby('duns_number').apply(
                lambda x: x.nlargest(topk, ['similarity'])).reset_index(drop=True)
        sample_pd.to_csv('sampled_' + pre_name + save_name )
    print('saving total data...')

    roc_auc = 0

    metrics = {}
    metrics['valid_f1'] = 0 #fbeta_score(all_targets, all_predictions, beta=1, average='macro')
    metrics['valid_loss'] = np.mean(all_losses)
    metrics['valid_top1'] = 0#acc[0].item()
    metrics['auc'] = roc_auc
    metrics['valid_top5'] = 0 #acc[1].item()

    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(metrics.items(), key=lambda kv: -kv[1])))

    return metrics

#=============================================================================================================================
#train
#=============================================================================================================================
def train(args, model: nn.Module, criterion, *, params,
          train_loader, valid_loader, init_optimizer, use_cuda,
          n_epochs=None, patience=2, max_lr_changes=3) -> bool:
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    params = list(params)#in case params is not a list
    #add params into optimizer
    optimizer = init_optimizer(params, lr)

    #model load/save path
    run_root = Path(args.run_root)

    model_path = Path(str(run_root) + '/' + 'model.pt')

    if model_path.exists():
        print('loading existing weights from model.pt')
        state = load_model(model, model_path)
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        best_f1 = state['best_f1']
    else:
        epoch = 1
        step = 0
        best_valid_loss = 0.0#float('inf')
        best_f1 = 0


    lr_changes = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss,
        'best_f1': best_f1
    }, str(model_path))

    save_where = lambda ep,svpath: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss,
        'best_f1': best_f1
    }, str(svpath))

    report_each = 100
    log = run_root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    valid_f1s = []
    lr_reset_epoch = epoch

    #epoch loop
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              (nPosTr+nNegTr)*len(train_loader) * args.batch_size))

        if epoch >= 20 and epoch%2==0:
            lr = lr * 0.9
            adjust_learning_rate(optimizer, lr)
            print('lr updated to %0.8f'%lr)

        tq.set_description('Epoch %d, lr %0.8f'%(epoch,lr))
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0

            for i, batch_dat in enumerate(tl):#enumerate() turns tl into index, ele_of_tl
                featComp = batch_dat['feat_comp']
                featLoc = batch_dat['feat_loc']
                featId = batch_dat['feat_id']
                featEnsemble = batch_dat['feat_ensemble_score']
                targets = batch_dat['target']

                # print(featComp.shape,featLoc.shape,batch_dat['feat_comp_dim'],batch_dat['feat_loc_dim'])

                if use_cuda:
                    featComp, featLoc, targets, featId,featEnsemble = featComp.cuda(), featLoc.cuda(),targets.cuda(),featId.cuda(),featEnsemble.cuda()

                # common_feat_comp, common_feat_loc, feat_comp_loc, outputs = model(feat_comp=featComp, feat_loc=featLoc)
                model_output = model(feat_comp = featComp, feat_loc = featLoc, id_loc = featId,feat_ensemble_score=featEnsemble)
                outputs = model_output['outputs']


                # outputs = outputs.squeeze()

                # loss1 = softmax_loss(outputs, targets)
                # loss2 = TripletLossV1(margin=0.5)(feats,targets)
                # loss1 = criterion(outputs,targets)


                if args.cos_sim_loss:
                    out_comp_feat = model_output['comp_feat']
                    out_loc_feat = model_output['loc_feat']
                    cos_targets = 2*targets.float()-1.0
                    loss = criterion(out_comp_feat,out_loc_feat,cos_targets)
                    lossType = 'cosine'
                else:
                    loss = softmax_loss(outputs, targets)
                    lossType = 'softmax'

                batch_size = featComp.size(0)

                (batch_size * loss).backward()
                if (i + 1) % args.step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                tq.update(batch_size)
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
                tq.set_postfix(loss=f'{mean_loss:.3f}')

                if i and i % report_each == 0:
                    write_event(log, step, loss=mean_loss)

            write_event(log, step, loss=mean_loss)
            tq.close()
            print('saving')
            save(epoch + 1)
            print('validation')
            valid_metrics = validation(model, criterion, valid_loader, use_cuda, lossType=lossType)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_top1 = valid_metrics['valid_top1']
            valid_roc = valid_metrics['auc']
            valid_losses.append(valid_loss)


            #tricky
            valid_loss = valid_roc
            if valid_loss > best_valid_loss:#roc:bigger is better
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(run_root) + '/model_loss_best.pt')

        except KeyboardInterrupt:
            tq.close()
            # print('Ctrl+C, saving snapshot')
            # save(epoch)
            # print('done.')

            return False
    return True

#=============================================================================================================================
#validation
#=============================================================================================================================
def validation(
        model: nn.Module, criterion, valid_loader, use_cuda, lossType='softmax', num_loc = 0, topK=0, Query_Company=True
        ) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for batch_dat in valid_loader:
            featComp = batch_dat['feat_comp']
            featLoc = batch_dat['feat_loc']
            featId = batch_dat['feat_id']
            targets = batch_dat['target']
            featEnsemble = batch_dat['feat_ensemble_score']
            all_targets.append(targets)#torch@cpu
            if use_cuda:
                featComp, featLoc, targets, featId,featEnsemble = featComp.cuda(), featLoc.cuda(), targets.cuda(), featId.cuda(),featEnsemble.cuda()
            model_output = model(feat_comp=featComp, feat_loc=featLoc, id_loc = featId,feat_ensemble_score=featEnsemble)
            outputs = model_output['outputs']
            # outputs = outputs.squeeze()
            # print(outputs.shape,targets.shape)
            # targets = targets.float()
            # loss = criterion(outputs, targets)
            # loss = softmax_loss(outputs, targets)

            if lossType=='softmax':
                loss = softmax_loss(outputs, targets)
                all_predictions.append(outputs)
            else:
                out_comp_feat = model_output['comp_feat']
                out_loc_feat = model_output['loc_feat']
                cos_targets = 2 * targets.float() - 1.0
                loss = criterion(out_comp_feat, out_loc_feat, cos_targets)
                all_predictions.append(model_output['outputs_cos'])


            # cur_batch_predictions = outputs[:, 1].data.cpu().numpy()
            # cur_batch_targets = targets.data.cpu().numpy()
            # fpr, tpr, roc_thresholds = roc_curve(cur_batch_targets, cur_batch_predictions)
            all_losses.append(loss.data.cpu().numpy())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)#list->torch
    print('all_predictions.shape: ')
    print(all_predictions.shape)

    if lossType=='softmax':
        all_predictions = F.softmax(all_predictions, dim=1)
        all_predictions2 = all_predictions[:, 1].data.cpu().numpy()
    else:
        all_predictions = (all_predictions + 1)/2 #squeeze to [0,1]
        all_predictions2 = all_predictions.data.cpu().numpy()

    # acc = topkAcc(all_predictions,all_targets.cuda(),topk=(1,))

    # value, all_predictions = all_predictions.topk(1, dim=1, largest=True, sorted=True)

    # all_predictions = all_predictions.data.cpu().numpy()
    all_targets =all_targets.data.cpu().numpy()

    # save_obj(all_targets,'all_targets')
    # save_obj(all_predictions2,'all_predictions2')

    fpr, tpr, roc_thresholds = roc_curve(all_targets, all_predictions2)

    roc_auc = auc(fpr,tpr)

    if topK > 0 and num_loc > 0:
        if Query_Company:#topk accuracy for company query
            all_predictions2 = all_predictions2.reshape(-1,num_loc)
            all_targets = all_targets.reshape(-1,num_loc)
            print( 'topk data reforming checking: ', (all_targets.sum(axis=1) == 1).all() )
            # truth_cat = one_hot_2_idx_numpy(all_targets)
            # R_cat = np.array(list(range(num_loc)))
            # topk_precision = calc_topk_acc_cat_all(all_predictions2, truth_cat, R_cat, k=topK)
        else:
            all_predictions2 = all_predictions2.reshape(num_loc, -1)
            all_targets = all_targets.reshape(num_loc, -1)

        topk_recall = topk_recall_score_all(pred=all_predictions2, truth=all_targets, topk=topK)

        step = int(topK / 10)
        x = list(range(1, topK + 1))
        y = list(topk_recall.reshape(-1))
        plt.figure()
        plt.plot(x, y)
        plt.grid()

        for z in range(10, topK + 1, step):
            z = z - 1
            plt.text(z, y[z], '%.4f' % y[z], ha='center', va='bottom', fontsize=9)

        plt.xlabel("topk")
        plt.ylabel("recall")

        if Query_Company:
            plt.title("topk recall curve of %d location" % num_loc)
            plt.savefig('topk_recall_of_company_query_%d.jpg' % num_loc)
        else:
            plt.title("topk recall curve of companies with %d locations" % num_loc)
            plt.savefig('topk_recall_of_location_query_%d.jpg' % num_loc)
        plt.close()


    metrics = {}
    metrics['valid_f1'] = 0 #fbeta_score(all_targets, all_predictions, beta=1, average='macro')
    metrics['valid_loss'] = np.mean(all_losses)
    metrics['valid_top1'] = 0#acc[0].item()
    metrics['auc'] = roc_auc
    metrics['valid_top5'] = 0 #acc[1].item()

    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(metrics.items(), key=lambda kv: -kv[1])))

    return metrics


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


