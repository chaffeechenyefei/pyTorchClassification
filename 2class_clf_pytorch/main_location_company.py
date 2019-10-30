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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

pjoin = os.path.join
#not used @this version
TR_DATA_ROOT = '/home/ubuntu/location_recommender_system/'
TT_DATA_ROOT = '/home/ubuntu/location_recommender_system/'

OLD_N_CLASSES = 2
N_CLASSES = 2#253#109
#=============================================================================================================================
#main
#=============================================================================================================================
def main():
    #cmd and arg parser
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', choices=['train', 'validate', 'predict_valid', 'predict_test'], default='train')
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
    arg('--n-epochs', type=int, default=120)
    arg('--epoch-size', type=int)
    arg('--tta', type=int, default=1)
    arg('--use-sample', action='store_true', help='use a sample of the dataset')
    arg('--debug', action='store_true')
    arg('--imgsize',type=int, default = 256)
    arg('--finetuning',action='store_true')
    arg('--cos_sim_loss',action='store_true')
    arg('--ensemble', action='store_true')
    arg('--sample_rate',type=float,default=0.5)

    #cuda version T/F
    use_cuda = cuda.is_available()

    args = parser.parse_args()
    #run_root: model/weights root
    run_root = Path(args.run_root)

    df_all_pair = pd.read_csv(pjoin(TR_DATA_ROOT,'train_val_test_location_company_all.csv'),index_col=0)
    df_comp_feat = pd.read_csv(pjoin(TR_DATA_ROOT,'company_feat.csv'),index_col=0)
    df_loc_feat = pd.read_csv(pjoin(TR_DATA_ROOT,'location_feat.csv'),index_col=0)


    if args.ensemble:
        df_ensemble = []
        ensemble_score = ['PA_comp_loc_score_exclude_self.csv','SJ_comp_loc_score_exclude_self.csv','SF_comp_loc_score_exclude_self.csv']
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
    del df_all_pair

    loc_name_dict = translocname2dict(df_loc_feat)

    ##::DataLoader
    def make_loader(df_comp_feat: pd.DataFrame, df_loc_feat: pd.DataFrame, df_pair: pd.DataFrame, emb_dict:dict,df_ensemble,
                    name='train',flag_ensemble=args.ensemble) -> DataLoader:
        return DataLoader(
            TrainDatasetLocationRS(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_pair,df_ensemble_score=df_ensemble,
                                   emb_dict=emb_dict, name=name,flag_ensemble=flag_ensemble,
                                   negN=200, posN=100),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=collate_TrainDatasetLocationRS
        )

    #Not used in this version
    # criterion = nn.BCEWithLogitsLoss(reduction='none')
    # criterion = nn.CrossEntropyLoss(reduction='none')
    criterion = nn.CosineEmbeddingLoss(0.2,reduce=True,reduction='mean')

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

    elif args.mode == 'validate':
        for ind_city in range(3):
            df_valid_pair_city = df_valid_pair[df_valid_pair['city']==ind_city]
            valid_loader = make_loader(df_comp_feat=df_comp_feat, df_loc_feat=df_loc_feat, df_pair=df_valid_pair_city,
                                   emb_dict=loc_name_dict,df_ensemble=df_ensemble, name='valid')
            print('Predictions for city %d'%ind_city)
            validation(model, criterion, tqdm.tqdm(valid_loader, desc='Validation'),
                       use_cuda=use_cuda)
        # if args.finetuning:
        #     pass
        # else:
        #     load_model(model, Path(str(run_root) + '/' + args.ckpt))
        # model.set_infer_mode()


#=============================================================================================================================
#End of main
#=============================================================================================================================

# #=============================================================================================================================
# #predict
# #=============================================================================================================================
# def predict(model, root: Path, df: pd.DataFrame, out_path: Path,
#             batch_size: int, tta_code:list , workers: int, use_cuda: bool):
#
#     loader = DataLoader(
#         dataset=TTADataset(root, df, tta_code=tta_code),
#         shuffle=False,
#         batch_size=batch_size,
#         num_workers=workers,
#     )
#
#     model.eval()
#     all_outputs, all_ids = [], []
#     with torch.no_grad():
#         for inputs, ids in tqdm.tqdm(loader, desc='Predict'):
#             if use_cuda:
#                 inputs = inputs.cuda()
#             outputs = torch.sigmoid(model(inputs))
#             #_, outputs = outputs.topk(1, dim=1, largest=True, sorted=True)
#             all_outputs.append(outputs.data.cpu().numpy())
#             all_ids.extend(ids)
#
#     df = pd.DataFrame(
#         data=np.concatenate(all_outputs),
#         index=all_ids,
#         columns=map(str, range(N_CLASSES)))
#
#     df = mean_df(df)
#     df.to_hdf(out_path, 'prob', index_label='id')
#     print('Saved predictions to %s' % out_path)

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
                              300*len(train_loader) * args.batch_size))

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

                loss1 = softmax_loss(outputs, targets)
                # loss2 = TripletLossV1(margin=0.5)(feats,targets)
                # loss1 = criterion(outputs,targets)


                if args.cos_sim_loss:
                    out_comp_feat = model_output['comp_feat']
                    out_loc_feat = model_output['loc_feat']
                    cos_targets = 2*targets.float()-1.0
                    loss2 = criterion(out_comp_feat,out_loc_feat,cos_targets)
                    loss = 0.8*loss1+0.2*loss2
                else:
                    loss = 1.0*loss1

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
            valid_metrics = validation(model, criterion, valid_loader, use_cuda)
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
        model: nn.Module, criterion, valid_loader, use_cuda,
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
            loss = softmax_loss(outputs, targets)

            # cur_batch_predictions = outputs[:, 1].data.cpu().numpy()
            # cur_batch_targets = targets.data.cpu().numpy()
            # fpr, tpr, roc_thresholds = roc_curve(cur_batch_targets, cur_batch_predictions)
            all_losses.append(loss.data.cpu().numpy())
            all_predictions.append(outputs)
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)#list->torch
    print('all_predictions.shape: ')
    print(all_predictions.shape)

    all_predictions2 = all_predictions[:, 1].data.cpu().numpy()

    # acc = topkAcc(all_predictions,all_targets.cuda(),topk=(1,))

    # value, all_predictions = all_predictions.topk(1, dim=1, largest=True, sorted=True)

    # all_predictions = all_predictions.data.cpu().numpy()
    all_targets =all_targets.data.cpu().numpy()

    fpr, tpr, roc_thresholds = roc_curve(all_targets, all_predictions2)
    roc_auc = auc(fpr,tpr)

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


