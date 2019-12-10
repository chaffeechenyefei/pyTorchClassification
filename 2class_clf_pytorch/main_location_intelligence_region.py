import os, sys

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
from dataset import TrainDatasetLocationRSRB, collate_TrainDatasetLocationRSRB, TestDatasetLocationRSRB, \
    collate_TestDatasetLocationRSRB
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from utils import (write_event, load_model, ThreadingDataLoader as DataLoader, adjust_learning_rate,
                   ON_KAGGLE)

from models.utils import *
from udf.basic import save_obj, load_obj, calc_topk_acc_cat_all, topk_recall_score_all
import matplotlib.pyplot as plt

# from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

pjoin = os.path.join
# not used @this version /home/ubuntu/location_recommender_system/ /Users/yefeichen/Database/location_recommender_system
TR_DATA_ROOT = '/home/ubuntu/location_recommender_system/'
TT_DATA_ROOT = '/home/ubuntu/location_recommender_system/'

OLD_N_CLASSES = 2
N_CLASSES = 2  # 253#109

nPosTr = 1000
nNegTr = 2000

model_name = ''  # same as main cmd --model XXX
wework_location_only = True


# =============================================================================================================================
# main
# =============================================================================================================================
def main():
    # cmd and arg parser
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', choices=['train', 'validate', 'predict'], default='validate')
    arg('--run_root', default='result/location_company_test')
    arg('--fold', type=int, default=0)
    arg('--model', default='location_recommend_region_model_v1')
    arg('--ckpt', type=str, default='model_loss_best.pt')
    arg('--batch-size', type=int, default=1)
    arg('--step', type=str, default=8)  # update the gradients every 8 batch(sample num = step*batch-size*inner_size)
    arg('--workers', type=int, default=16)
    arg('--lr', type=float, default=3e-4)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=80)
    arg('--epoch-size', type=int)
    arg('--finetuning', action='store_true')
    arg('--testStep', type=int, default=50000)
    arg('--trainStep', type=int, default=10000)
    arg('--citynum', type=int, default=5)
    arg('--apps', type=str, default='_191114.csv')

    # cuda version T/F
    use_cuda = cuda.is_available()

    args = parser.parse_args()
    # run_root: model/weights root
    run_root = Path(args.run_root)

    global model_name
    model_name = args.model

    if args.mode in ['train', 'validate']:
        df_train_pair = pd.read_csv(pjoin(TR_DATA_ROOT, 'region_train' + args.apps), index_col=0)
        print('num of train pair %d' % len(df_train_pair))
        df_valid_pair = pd.read_csv(pjoin(TR_DATA_ROOT, 'region_test' + args.apps), index_col=0)
        print('num of valid pair %d' % len(df_valid_pair))
    elif args.mode in ['predict']:
        c_salesforce_file = 'salesforce_comp_city_from_opp.csv'

    df_comp_feat = pd.read_csv(pjoin(TR_DATA_ROOT, 'company_feat' + args.apps), index_col=0)
    df_loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT, 'location_feat' + args.apps), index_col=0)

    citynameabbr = ['PA', 'SF', 'SJ', 'LA', 'NY']
    cityname = ['Palo Alto', 'San Francisco', 'San Jose', 'Los Angeles', 'New York']

    cfile = ['dnb_pa.csv', 'dnb_sf.csv', 'dnb_sj.csv', 'dnb_Los_Angeles.csv', 'dnb_New_York.csv']
    lfile = 'location_scorecard_191113.csv'

    clfile = [c + args.apps for c in citynameabbr]
    pred_save_name = [c + '_similarity' + args.apps for c in citynameabbr]

    ##::DataLoader
    if args.mode in ['train', 'validate']:
        def make_loader(df_comp_feat: pd.DataFrame, df_loc_feat: pd.DataFrame, df_pair: pd.DataFrame, trainStep=10000,
                        name='train', shuffle=True) -> DataLoader:
            return DataLoader(
                TrainDatasetLocationRSRB(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, name=name, df_pair=df_pair,
                                         citynum=args.citynum, trainStep=trainStep),
                shuffle=shuffle,
                batch_size=args.batch_size,
                num_workers=args.workers,
                collate_fn=collate_TrainDatasetLocationRSRB
            )
    elif args.mode in ['predict']:
        """
        loading embedded feature vector
        """
        df_region_feat = pd.read_csv(pjoin(TR_DATA_ROOT,'location_feat_emb_'+args.model+'.csv'),index_col=0)
        def make_loader(df_comp_feat: pd.DataFrame, df_loc_feat: pd.DataFrame, df_region_feat: pd.DataFrame,
                        df_pair: pd.DataFrame, trainStep=10000,
                        name='predict', shuffle=False) -> DataLoader:
            return DataLoader(
                TestDatasetLocationRSRB(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat,
                                        df_region_feat=df_region_feat, df_pair=df_pair, citynum=args.citynum,
                                        trainStep=trainStep),
                shuffle=shuffle,
                batch_size=args.batch_size,
                num_workers=args.workers,
                collate_fn=collate_TestDatasetLocationRSRB
            )

    # Not used in this version
    # criterion = nn.BCEWithLogitsLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = softmax_loss
    lossType = 'softmax'

    # se- ception dpn can only use finetuned model from imagenet
    model = getattr(rsmodels, args.model)(feat_comp_dim=102)  # location_recommend_model_v3

    md_path = Path(str(run_root) + '/' + args.ckpt)
    if md_path.exists():
        print('load weights from md_path')
        load_model(model, md_path)

    ##params::Add here
    # all_params = list(model.parameters())
    all_params = filter(lambda p: p.requires_grad, model.parameters())

    # apply parallel gpu if available
    # model = torch.nn.DataParallel(model)

    # gpu first
    if use_cuda:
        model = model.cuda()

    # print(model)
    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        Path(str(run_root) + '/params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        train_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_train_pair,
                                   name='train_fast', trainStep=args.trainStep)
        valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_valid_pair,
                                   name='valid', shuffle=False)

        train_kwargs = dict(
            args=args,
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            patience=args.patience,
            init_optimizer=lambda params, lr: Adam(params, lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=2e-4),
            use_cuda=use_cuda,
        )

        train(params=all_params, **train_kwargs)

    elif args.mode == 'validate':
        """
        """
        valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_valid_pair,
                                   name='valid', shuffle=False)
        validation(args, model, criterion, tqdm.tqdm(valid_loader, desc='Validation'), use_cuda=use_cuda,
                   lossType=lossType)

    elif args.mode == 'predict':
        """
        all region features are embedded as vectors before ahead
        """
        pdc_all = pd.read_csv(pjoin(TR_DATA_ROOT, c_salesforce_file))[['duns_number', 'city']]
        for ind_city, str_city in enumerate(cityname):
            pdcl = pd.read_csv(pjoin(TR_DATA_ROOT, clfile[ind_city]))[['atlas_location_uuid', 'duns_number']]
            pdc = pdc_all[pdc_all['city'] == str_city]
            tot_comp = len(pdc)
            print('Total %d company found in %s from salesforce' % (tot_comp, str_city))
            pdc['atlas_location_uuid'] = 'a'
            # in case of multi-mapping
            all_loc_name = pdcl[['atlas_location_uuid']].groupby(['atlas_location_uuid'])[
                ['atlas_location_uuid']].first().reset_index(drop=True)

            if wework_location_only:
                loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT, lfile))[['atlas_location_uuid', 'is_wework']]
                loc_ww = loc_feat[loc_feat['is_wework'] == True]
                all_loc_name = \
                    all_loc_name.merge(loc_ww, on='atlas_location_uuid', how='inner', suffixes=['', '_right'])[
                        ['atlas_location_uuid']]

            tot_loc = len(all_loc_name)
            print('Total %d locations belonged to ww in %s' % (tot_loc, str_city))

            all_loc_name['key'] = 0
            pdc['key'] = 0

            testing_pair = pd.merge(pdc, all_loc_name, on='key', how='left',
                                    suffixes=['_left', '_right']).reset_index(drop=True)

            testing_pair = testing_pair.rename(
                columns={'atlas_location_uuid_left': 'groundtruth', 'atlas_location_uuid_right': 'atlas_location_uuid'})
            testing_pair = testing_pair[['duns_number', 'atlas_location_uuid', 'groundtruth']]
            testing_pair['label'] = (testing_pair['atlas_location_uuid'] == testing_pair['groundtruth'])
            testing_pair = testing_pair[['duns_number', 'atlas_location_uuid', 'label']]

            predict_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat,
                                         df_region_feat=df_region_feat, df_pair=testing_pair, name='predict',
                                         shuffle=False)

            print('Predictions for city %d' % ind_city)

            if wework_location_only:
                pre_name = 'ww_'
            else:
                pre_name = ''

            if args.query_location:
                topk = min(300, tot_comp)
            else:
                topk = min(3, tot_loc)

            predict(model, criterion, tqdm.tqdm(predict_loader, desc='Prediction'),
                    use_cuda=use_cuda, test_pair=testing_pair[['atlas_location_uuid', 'duns_number']],
                    pre_name=pre_name, \
                    save_name=pred_save_name[ind_city], lossType=lossType, query_loc_flag=args.query_location,
                    topk=topk)


# =============================================================================================================================
# End of main
# =============================================================================================================================
# =============================================================================================================================
# train
# =============================================================================================================================
def train(args, model: nn.Module, criterion, *, params,
          train_loader, valid_loader, init_optimizer, use_cuda,
          n_epochs=None, patience=2, max_lr_changes=3) -> bool:
    lr = args.lr
    n_epochs = n_epochs or args.n_epochs
    params = list(params)  # in case params is not a list
    # add params into optimizer
    optimizer = init_optimizer(params, lr)

    # model load/save path
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
        best_valid_loss = 0.0  # float('inf')
        best_f1 = 0

    lr_changes = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss,
        'best_f1': best_f1
    }, str(model_path))

    save_where = lambda ep, svpath: torch.save({
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

    # epoch loop
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              len(train_loader) * args.batch_size))

        if epoch >= 20 and epoch % 2 == 0:
            lr = lr * 0.9
            adjust_learning_rate(optimizer, lr)
            print('lr updated to %0.8f' % lr)

        tq.set_description('Epoch %d, lr %0.8f' % (epoch, lr))
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0

            for i, batch_dat in enumerate(tl):  # enumerate() turns tl into index, ele_of_tl
                featCompPos = batch_dat['feat_comp_pos']
                featCompNeg = batch_dat['feat_comp_neg']
                featRegion = batch_dat['feat_comp_region']
                featLoc = batch_dat['feat_loc']

                if use_cuda:
                    featCompPos, featCompNeg, featRegion, featLoc = featCompPos.cuda(), featCompNeg.cuda(), featRegion.cuda(), featLoc.cuda()

                # common_feat_comp, common_feat_loc, feat_comp_loc, outputs = model(feat_comp=featComp, feat_loc=featLoc)
                if args.model == 'location_recommend_region_model_v1':
                    model_output_pos = model(feat_comp=featCompPos, feat_K_comp=featRegion)
                    model_output_neg = model(feat_comp=featCompNeg, feat_K_comp=featRegion)
                elif args.model == 'location_recommend_region_model_v0':
                    model_output_pos = model(feat_comp=featCompPos, feat_loc=featLoc)
                    model_output_neg = model(feat_comp=featCompNeg, feat_loc=featLoc)
                else:
                    model_output_pos = model(feat_comp=featCompPos, feat_K_comp=featRegion, feat_loc=featLoc)
                    model_output_neg = model(feat_comp=featCompNeg, feat_K_comp=featRegion, feat_loc=featLoc)

                # outputs = torch.cat( [ model_output_pos['outputs'], model_output_neg['outputs'] ], dim = 0)

                nP, nN = model_output_pos['outputs'].shape[0], model_output_neg['outputs'].shape[0]
                target_pos = torch.ones((nP, 1), dtype=torch.long)
                target_neg = torch.zeros((nN, 1), dtype=torch.long)
                # targets = torch.cat( [ target_pos, target_neg ], dim = 0)

                if use_cuda:
                    # targets = targets.cuda()
                    target_pos = target_pos.cuda()
                    target_neg = target_neg.cuda()

                lossP = softmax_loss(model_output_pos['outputs'], target_pos)
                lossN = softmax_loss(model_output_neg['outputs'], target_neg)
                loss = lossP + 0.8 * lossN

                if args.model in ['location_recommend_region_model_v4', 'location_recommend_region_model_v5']:
                    lW = 0.1
                    loss2 = l2_loss(model_output_pos['feat_loc_pred'], featLoc)  # Neg Pos share the same location
                    loss = (1 - lW) * loss + lW * loss2

                lossType = 'softmax'

                batch_size = nP + nN

                (batch_size * loss).backward()
                if (i + 1) % args.step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                tq.update(1 * args.batch_size)
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
            valid_metrics = validation(args, model, criterion, valid_loader, use_cuda, lossType=lossType)
            write_event(log, step, **valid_metrics)
            valid_loss = valid_metrics['valid_loss']
            valid_top1 = valid_metrics['valid_top1']
            valid_roc = valid_metrics['auc']
            valid_losses.append(valid_loss)

            # tricky
            valid_loss = valid_roc
            if valid_loss > best_valid_loss:  # roc:bigger is better
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(run_root) + '/model_loss_best.pt')

        except KeyboardInterrupt:
            tq.close()
            # print('Ctrl+C, saving snapshot')
            # save(epoch)
            # print('done.')

            return False
    return True


# =============================================================================================================================
# validation
# =============================================================================================================================
def validation(
        args, model: nn.Module, criterion, valid_loader, use_cuda, lossType='softmax') -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for batch_dat in valid_loader:
            featCompPos = batch_dat['feat_comp_pos']
            featCompNeg = batch_dat['feat_comp_neg']
            featRegion = batch_dat['feat_comp_region']
            featLoc = batch_dat['feat_loc']

            if use_cuda:
                featCompPos, featCompNeg, featRegion, featLoc = featCompPos.cuda(), featCompNeg.cuda(), featRegion.cuda(), featLoc.cuda()

            # common_feat_comp, common_feat_loc, feat_comp_loc, outputs = model(feat_comp=featComp, feat_loc=featLoc)
            if args.model == 'location_recommend_region_model_v1':
                model_output_pos = model(feat_comp=featCompPos, feat_K_comp=featRegion)
                model_output_neg = model(feat_comp=featCompNeg, feat_K_comp=featRegion)
            elif args.model == 'location_recommend_region_model_v0':
                model_output_pos = model(feat_comp=featCompPos, feat_loc=featLoc)
                model_output_neg = model(feat_comp=featCompNeg, feat_loc=featLoc)
            else:
                model_output_pos = model(feat_comp=featCompPos, feat_K_comp=featRegion, feat_loc=featLoc)
                model_output_neg = model(feat_comp=featCompNeg, feat_K_comp=featRegion, feat_loc=featLoc)

            outputs = torch.cat([model_output_pos['outputs'], model_output_neg['outputs']], dim=0)

            nP, nN = model_output_pos['outputs'].shape[0], model_output_neg['outputs'].shape[0]
            target_pos = torch.ones((nP, 1), dtype=torch.long)
            target_neg = torch.zeros((nN, 1), dtype=torch.long)
            targets = torch.cat([target_pos, target_neg], dim=0)

            if use_cuda:
                targets = targets.cuda()

            loss = softmax_loss(outputs, targets)

            all_predictions.append(outputs)
            all_targets.append(targets)
            all_losses.append(loss.data.cpu().numpy())

    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)  # list->torch
    print('all_predictions.shape: ')
    print(all_predictions.shape)

    if lossType == 'softmax':
        all_predictions = F.softmax(all_predictions, dim=1)
        all_predictions2 = all_predictions[:, 1].data.cpu().numpy()
    else:
        all_predictions = (all_predictions + 1) / 2  # squeeze to [0,1]
        all_predictions2 = all_predictions.data.cpu().numpy()

    all_targets = all_targets.data.cpu().numpy()

    # save_obj(all_targets,'all_targets')
    # save_obj(all_predictions2,'all_predictions2')

    fpr, tpr, roc_thresholds = roc_curve(all_targets, all_predictions2)

    roc_auc = auc(fpr, tpr)

    metrics = {}
    metrics['valid_f1'] = 0  # fbeta_score(all_targets, all_predictions, beta=1, average='macro')
    metrics['valid_loss'] = np.mean(all_losses)
    metrics['valid_top1'] = 0  # acc[0].item()
    metrics['auc'] = roc_auc
    metrics['valid_top5'] = 0  # acc[1].item()

    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(metrics.items(), key=lambda kv: -kv[1])))

    return metrics


# =============================================================================================================================
# prediction
# =============================================================================================================================
def predict(
        args, model: nn.Module, criterion, predict_loader, use_cuda, test_pair, save_name: str, pre_name: str = '',
        topk=300, query_loc_flag=True, sampling=True) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for batch_dat in predict_loader:
            featComp = batch_dat['feat_comp']
            featRegion = batch_dat['feat_region']
            featLoc = batch_dat['feat_loc']
            targets = batch_dat['targets']

            if use_cuda:
                featComp, featLoc, targets, featRegion = featComp.cuda(), featLoc.cuda(), targets.cuda(), featRegion.cuda()

            model_output = model.predict(feat_comp=featComp, feat_region=featRegion, feat_loc=featLoc)

            outputs = model_output['outputs']

            loss = softmax_loss(outputs, targets)

            all_predictions.append(outputs)
            all_targets.append(targets)
            all_losses.append(loss.data.cpu().numpy())
    # end_with


    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)  # list->torch
    print('all_predictions.shape: ')
    print(all_predictions.shape)

    all_predictions = F.softmax(all_predictions, dim=1)
    all_predictions2 = all_predictions[:, 1].data.cpu().numpy()

    print('saving...')
    dat_pred_pd = pd.DataFrame(data=all_predictions2.reshape(-1, 1), columns=['similarity'])
    res_pd = pd.concat([test_pair, dat_pred_pd], axis=1)

    if sampling:
        print('sampling...')
        # for each location we return topk companies
        if query_loc_flag:
            sample_pd = res_pd.groupby('atlas_location_uuid').apply(
                lambda x: x.nlargest(topk, ['similarity'])).reset_index(drop=True)
        else:
            sample_pd = res_pd.groupby('duns_number').apply(
                lambda x: x.nlargest(topk, ['similarity'])).reset_index(drop=True)
        sample_pd.to_csv('sampled_' + pre_name + save_name)
    print('saving total data...')

    roc_auc = 0

    metrics = {}
    metrics['valid_f1'] = 0  # fbeta_score(all_targets, all_predictions, beta=1, average='macro')
    metrics['valid_loss'] = np.mean(all_losses)
    metrics['valid_top1'] = 0  # acc[0].item()
    metrics['auc'] = roc_auc
    metrics['valid_top5'] = 0  # acc[1].item()

    print(' | '.join(f'{k} {v:.3f}' for k, v in sorted(metrics.items(), key=lambda kv: -kv[1])))

    return metrics


def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]


def softmax_loss(results, labels):
    labels = labels.view(-1)
    loss = F.cross_entropy(results, labels, reduce=True)

    return loss


def l2_loss(results, targets):
    diff = results - targets
    loss = torch.sum(diff ** 2, dim=1, keepdim=True)
    loss = loss.mean()

    return loss


def softmax_lossV2(results, labels):
    softmax_label = labels[labels < N_CLASSES].view(-1)
    label_len = softmax_label.shape[0]
    softmax_results = results[:label_len, :]
    assert (label_len % 2 == 0)
    loss = F.cross_entropy(softmax_results, softmax_label, reduce=True)

    return loss


if __name__ == '__main__':
    main()
