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
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch import nn, cuda
from torch.optim import Adam, SGD
import tqdm
import os
import models.models as models
from dataset import TrainDataset, TTADataset, get_ids, N_CLASSES,OLD_N_CLASSES, DATA_ROOT,collate_TrainDatasetTriplet,TrainDatasetTriplet
from transforms import train_transform, test_transform
from utils import (write_event, load_model, load_par_gpu_model_gpu, mean_df, ThreadingDataLoader as DataLoader,
                   ON_KAGGLE)

from models.utils import *
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


#=============================================================================================================================
#main
#=============================================================================================================================
def main():
    #cmd and arg parser
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', choices=['train', 'validate', 'predict_valid', 'predict_test'], default='train')
    arg('--run_root', default='result/furniture_toy')
    arg('--fold', type=int, default=0)
    arg('--model', default='resnet50V4')
    arg('--ckpt', type=str, default='model_loss_best.pt')
    arg('--pretrained', type=str, default='imagenet')#resnet 1, resnext imagenet
    arg('--batch-size', type=int, default=8)
    arg('--step', type=str, default=8)
    arg('--workers', type=int, default=16)
    arg('--lr', type=float, default=3e-4)
    arg('--patience', type=int, default=4)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=60)
    arg('--epoch-size', type=int)
    arg('--tta', type=int, default=1)
    arg('--use-sample', action='store_true', help='use a sample of the dataset')
    arg('--debug', action='store_true')
    arg('--limit', type=int)
    arg('--imgsize',type=int, default = 256)
    arg('--finetuning',action='store_true')

    #cuda version T/F
    use_cuda = cuda.is_available()

    args = parser.parse_args()
    #run_root: model/weights root
    run_root = Path(args.run_root)
    #csv for train/test/validate [id,attribute_id,fold,data]
    folds = pd.read_csv('train_val_test_chair.csv')

    #Not used @this version...
    train_root = DATA_ROOT
    valid_root = DATA_ROOT

    # #Only images in train_sample are used folds = folds[bool vec]
    # if args.use_sample:
    #     folds = folds[folds['Id'].isin(set(get_ids(train_root)))]

    #split train/valid fold
    train_fold = folds[folds['fold'] == 0]
    valid_fold = folds[folds['fold'] == 1]

    #limit the size of train/valid data
    #W::Do not use it because the limited size of training data may not contain whole class
    if args.limit:
        train_fold = train_fold[:args.limit]
        valid_fold = valid_fold[:args.limit]

    ##::DataLoader
    def make_loader(df: pd.DataFrame, root, image_transform, name='train') -> DataLoader:
        if name == 'train':
            return DataLoader(
                TrainDatasetTriplet(root, df, debug=args.debug, name=name, imgsize = args.imgsize),
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=args.workers,
                collate_fn= collate_TrainDatasetTriplet
            )
        else:
            return DataLoader(
                TrainDataset(root, df, debug=args.debug, name=name, imgsize = args.imgsize),
                shuffle=True,
                batch_size=args.batch_size,
                num_workers=args.workers,
            )

    #Not used in this version
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    # criterion = nn.CrossEnropyLoss(reduction='none)

    # se- ception dpn can only use finetuned model from imagenet
    if args.finetuning:
        base_model_class = OLD_N_CLASSES
    else:
        base_model_class = N_CLASSES

    if 'se' not in args.model and 'ception' not in args.model and 'dpn' not in args.model:
        # model=> models.py
        model = getattr(models, args.model)(
            num_classes=base_model_class, pretrained=args.pretrained)
    else:
        model = getattr(models, args.model)(
            num_classes=base_model_class, pretrained='imagenet')


    if 'se' not in args.model and 'ception' not in args.model and 'dpn' not in args.model:
        fresh_params = list(model.fresh_params())



    #finetune::load model with old settings first and then change the last layer for new task!
    if args.finetuning:
        print('Doing finetune initial...')
        load_par_gpu_model_gpu(model, Path(str(run_root) + '/' + 'model_base.initial') )
        model.finetuning(N_CLASSES)

    ##params::Add here
    #params list[models.parameters()]
    all_params = list(model.parameters())

    #apply parallel gpu if available
    model = torch.nn.DataParallel(model)

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

        train_loader = make_loader(train_fold, train_root, train_transform, name='train')
        valid_loader = make_loader(valid_fold, valid_root, test_transform, name='valid')

        print(f'{len(train_loader.dataset):,} items in train, '
              f'{len(valid_loader.dataset):,} in valid')

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

        # #if args.pretrained and args.model != 'se_resnext50' and args.model != 'se_resnext101':
        # if 'se' not in args.model and 'ception' not in args.model and args.pretrained and 'dpn' not in args.model:
        #     if train(params=fresh_params, n_epochs=1, **train_kwargs):
        #         train(params=all_params, **train_kwargs)
        # else:
        #     train(params=all_params, **train_kwargs)


    elif args.mode == 'validate':
        valid_loader = make_loader(valid_fold, valid_root ,image_transform=test_transform, name='valid')
        load_model(model, Path(str(run_root) + '/' + args.ckpt))
        validation(model, criterion, tqdm.tqdm(valid_loader, desc='Validation'),
                   use_cuda=use_cuda)

    elif args.mode.startswith('predict'):

        load_model(model, Path(str(run_root) + '/' + args.ckpt))
        predict_kwargs = dict(
            batch_size=args.batch_size,
            tta=args.tta,
            use_cuda=use_cuda,
            workers=args.workers,
        )

        if args.mode == 'predict_valid':
            predict(model, df=valid_fold, root=train_root,
                    out_path=Path(str(run_root) + '/' + 'val.h5'),
                    **predict_kwargs)

        elif args.mode == 'predict_test':

            load_model(model, Path(str(run_root) + '/' + args.ckpt))
            print(args.ckpt)


            test_root = DATA_ROOT + '/' + (
                'test_sample' if args.use_sample else 'test2')
            test_df = pd.read_csv('train_val_test.csv')
            test_df = test_df[test_df['fold'] == 2]
            tta_code_list = []
            tta_code_list.append([0, 0])
            tta_code_list.append([0, 1])
            tta_code_list.append([0, 2])
            tta_code_list.append([0, 3])
            tta_code_list.append([0, 4])
            tta_code_list.append([1, 0])
            tta_code_list.append([1, 1])
            tta_code_list.append([1, 2])
            tta_code_list.append([1, 3])
            tta_code_list.append([1, 4])

            tta_code_list.append([0, 5])
            tta_code_list.append([1, 5])

            save_dir = str(run_root) + '/12tta'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for tta_code in tta_code_list:
                print(tta_code)
                predict(model,
                        df=test_df,
                        root=test_root,
                        out_path=Path(str(run_root) +  '/12tta/fold' + str(args.fold)+'_'+str(tta_code[0]) + str(tta_code[1]) + '_test.h5'),
                        batch_size = args.batch_size,
                        tta_code=tta_code,
                        workers=8,
                        use_cuda=True)
#=============================================================================================================================
#End of main
#=============================================================================================================================

#=============================================================================================================================
#predict
#=============================================================================================================================
def predict(model, root: Path, df: pd.DataFrame, out_path: Path,
            batch_size: int, tta_code:list , workers: int, use_cuda: bool):

    loader = DataLoader(
        dataset=TTADataset(root, df, tta_code=tta_code),
        shuffle=False,
        batch_size=batch_size,
        num_workers=workers,
    )

    model.eval()
    all_outputs, all_ids = [], []
    with torch.no_grad():
        for inputs, ids in tqdm.tqdm(loader, desc='Predict'):
            if use_cuda:
                inputs = inputs.cuda()
            outputs = torch.sigmoid(model(inputs))
            #_, outputs = outputs.topk(1, dim=1, largest=True, sorted=True)
            all_outputs.append(outputs.data.cpu().numpy())
            all_ids.extend(ids)

    df = pd.DataFrame(
        data=np.concatenate(all_outputs),
        index=all_ids,
        columns=map(str, range(N_CLASSES)))

    df = mean_df(df)
    df.to_hdf(out_path, 'prob', index_label='id')
    print(f'Saved predictions to {out_path}')

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
        state = load_model(model, model_path)
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        best_f1 = state['best_f1']
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')
        best_f1 = 0


    lr_changes = 0

    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep,
        'step': step,
        'best_valid_loss': best_valid_loss,
        'best_f1': best_f1
    }, str(model_path))

    report_each = 100
    log = run_root.joinpath('train.log').open('at', encoding='utf8')
    valid_losses = []
    valid_f1s = []
    lr_reset_epoch = epoch

    #epoch loop
    for epoch in range(epoch, n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(args.epoch_size or
                              TrainDatasetTriplet.tbatch()*len(train_loader) * args.batch_size))

        if epoch >= 10:
            lr = lr * 0.9
            optimizer = init_optimizer(params, lr)
            print(f'lr updated to {lr}')

        tq.set_description(f'Epoch {epoch}, lr {lr}')
        losses = []
        tl = train_loader
        if args.epoch_size:
            tl = islice(tl, args.epoch_size // args.batch_size)
        try:
            mean_loss = 0

            for i, (inputs, targets) in enumerate(tl):#enumerate() turns tl into index, ele_of_tl
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                feats, outputs= model(inputs)
                outputs = outputs.squeeze()
                feats = feats.squeeze()

                loss1 = softmax_loss(outputs, targets)
                loss2 = TripletLossV1(margin=0.5)(feats,targets)

                loss = 0.5*loss1 + loss2

                batch_size = inputs.size(0)

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
            valid_losses.append(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                shutil.copy(str(model_path), str(run_root) + '/model_loss_best.pt')

        except KeyboardInterrupt:
            tq.close()
            # print('Ctrl+C, saving snapshot')
            # save(epoch)
            # print('done.')

            return False
    return True

def validation(
        model: nn.Module, criterion, valid_loader, use_cuda,
        ) -> Dict[str, float]:
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets)#torch@cpu
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            _,outputs = model(inputs)#torch@gpu
            # outputs = outputs.squeeze()
            # targets = targets.float()
            # loss = criterion(outputs, targets)
            loss = softmax_loss(outputs, targets)
            all_losses.append(loss.data.cpu().numpy())
            all_predictions.append(outputs)
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)#list->torch

    acc = topkAcc(all_predictions,all_targets.cuda(),topk=(1,5))

    value, all_predictions = all_predictions.topk(1, dim=1, largest=True, sorted=True)

    all_predictions = all_predictions.data.cpu().numpy()
    all_targets =all_targets.data.cpu().numpy()

    metrics = {}
    metrics['valid_f1'] = fbeta_score(all_targets, all_predictions, beta=1, average='macro')
    metrics['valid_loss'] = np.mean(all_losses)
    metrics['valid_top1'] = acc[0].item()
    metrics['valid_top5'] = acc[1].item()
    #         all_losses.append(_reduce_loss(loss).item())
    #         predictions = torch.sigmoid(outputs)
    #         all_predictions.append(predictions.cpu().numpy())
    # all_predictions = np.concatenate(all_predictions)
    # all_targets = np.concatenate(all_targets)
    # metrics = {}
    # metrics['Recall'] = recall_score(all_targets, all_predictions)
    # metrics['Precision'] = precision_score(all_targets, all_predictions)
    # metrics['F1 score'] = f1_score(all_targets, all_predictions)
    # metrics['AUC'] = roc_auc_score(all_targets, all_predictions)
    # metrics['valid_loss'] = np.mean(all_losses)
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


