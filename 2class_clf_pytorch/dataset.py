from pathlib import Path
from typing import Callable, List
import random
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip)
from transforms import tensor_transform
from aug import *
from transforms import iaaTransform
import math
from udf.basic import timer

# image_size = 256

iaa_transformer = iaaTransform()
iaa_transformer.getSeq()

sfx = ['','_right']

# =======================================================================================================================
# Standard Data load with one image each time and do not consider triplet loss(P/N sample pairs)
# =======================================================================================================================
class TrainDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, debug: bool = True, name: str = 'train', imgsize=256,
                 class_num=-1):
        super().__init__()
        self._root = root
        self._df = df
        self._debug = debug
        self._name = name
        self._imgsize = imgsize
        self._class_num = class_num

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        _idx = idx
        item = self._df.iloc[_idx]
        # image = load_transform_image(item, self._root, imgsize = self._imgsize,debug=self._debug, name=self._name)
        image = load_transform_image_iaa(item, self._root, imgsize=self._imgsize, debug=self._debug, name=self._name)
        # target = torch.zeros(N_CLASSES)
        lb = item.attribute_ids
        # print(lb)

        # for cls in range(N_CLASSES):
        #     target[cls] = int(lb[cls + 1])
        # clsval = int(lb[5])
        # target = torch.from_numpy(np.array(item.attribute_ids))
        clsval = int(lb)
        assert (clsval >= 0 and clsval < self._class_num)
        target = torch.from_numpy(np.array(clsval))
        return image, target


# =======================================================================================================================
# Vanilla Data loader for Triplet loss, data loading is not fast because each time 8 images will be loaded and doing aug
# separately(not in batch version)
# =======================================================================================================================
class TrainDatasetTriplet(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, debug: bool = True, name: str = 'train', imgsize=256,
                 class_num=-1):
        super().__init__()
        self._root = root
        self._df = df
        self._debug = debug
        self._name = name
        self._imgsize = imgsize
        self._class_num = class_num

    def __len__(self):  # how much times will each epoch sample
        return min(max(len(self._df), 20000), 40000)
        # return self._class_num*125

    @staticmethod
    def tbatch():
        return 8

    def __getitem__(self, idx: int):
        # choose label from data a
        # choose any tow sample from data b bcz 1 image per class
        labelA = int(idx % self._class_num)
        dfA = self._df[self._df['attribute_ids'] == labelA]
        while dfA.empty:
            labelA = random.randint(0, self._class_num - 1)
            dfA = self._df[self._df['attribute_ids'] == labelA]

        len_dfA = len(dfA)
        assert (len_dfA != 0)
        pair_idxA = [random.randint(0, len_dfA - 1) for _ in range(4)]  # 有重采样
        images = []
        targets = []

        # pos
        for idxA in pair_idxA:
            item = dfA.iloc[idxA]
            image = load_transform_image_iaa(item, self._root, imgsize=self._imgsize, debug=self._debug,
                                             name=self._name)
            lb = int(item.attribute_ids)
            assert (lb < self._class_num)
            images.append(image)
            targets.append(lb)

        # neg
        dfB = self._df[self._df['attribute_ids'] != labelA]
        len_dfB = len(dfB)
        pair_idxB = [random.randint(0, len_dfB - 1) for _ in range(4)]  # 有重采样

        for idxB in pair_idxB:
            item = dfB.iloc[idxB]
            image = load_transform_image_iaa(item, self._root, imgsize=self._imgsize, debug=self._debug,
                                             name=self._name)
            images.append(image)
            lb = int(item.attribute_ids)
            targets.append(lb)

        return images, targets


def collate_TrainDatasetTriplet(batch):
    """
    special collate_fn function for UDF class TrainDatasetTriplet
    :param batch: 
    :return: 
    """
    # batch_size = len(batch)
    images = []
    labels = []

    for b in batch:
        if b[0] is None:
            continue
        else:
            images.extend(b[0])
            labels.extend(b[1])

    assert (len(images) == len(labels))

    images = torch.stack(images, 0)  # images : list of [C,H,W] -> [Len_of_list, C, H,W]
    labels = torch.from_numpy(np.array(labels))
    return images, labels


# =======================================================================================================================
# Standard Triplet loss data loader and data aug is done with batch
# =======================================================================================================================
class TrainDatasetTripletBatchAug(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, debug: bool = True, name: str = 'train', imgsize=256,
                 class_num=-1):
        super().__init__()
        self._root = root
        self._df = df
        self._debug = debug
        self._name = name
        self._imgsize = imgsize
        self._class_num = class_num

    def __len__(self):  # how much times will each epoch sample
        return min(max(len(self._df), 40000), 20000)
        # return self._class_num*125

    @staticmethod
    def tbatch():
        return 8

    def __getitem__(self, idx: int):
        # choose label from data a
        # choose any tow sample from data b bcz 1 image per class
        labelA = int(idx % self._class_num)

        dfA = self._df[self._df['attribute_ids'] == labelA]
        while dfA.empty:
            labelA = random.randint(0, self._class_num - 1)
            dfA = self._df[self._df['attribute_ids'] == labelA]

        len_dfA = len(dfA)
        assert (len_dfA != 0)
        pair_idxA = [random.randint(0, len_dfA - 1) for _ in range(4)]  # 有重采样

        images = []
        targets = []

        # pos
        for idxA in pair_idxA:
            item = dfA.iloc[idxA]
            image = load_image_uint8(item, self._root, imgsize=self._imgsize, debug=self._debug,
                                     name=self._name)
            lb = int(item.attribute_ids)
            assert (lb < self._class_num)
            images.append(image)
            targets.append(lb)

        # neg
        dfB = self._df[self._df['attribute_ids'] != labelA]
        len_dfB = len(dfB)
        pair_idxB = [random.randint(0, len_dfB - 1) for _ in range(4)]  # 有重采样

        for idxB in pair_idxB:
            item = dfB.iloc[idxB]
            image = load_image_uint8(item, self._root, imgsize=self._imgsize, debug=self._debug,
                                     name=self._name)
            images.append(image)
            lb = int(item.attribute_ids)
            targets.append(lb)

        # IAA Batch Operations
        images = np.stack(images, 0)

        # images = [B,H,W,C]
        images = iaa_transformer.act_batch(images)
        images = np.transpose(images, (0, 3, 1, 2))  # [B,H,W,C] -> [B,C,H,W]
        images = images.astype(np.float32)
        images = images / 255.0

        return torch.FloatTensor(images), targets


def collate_TrainDatasetTripletBatchAug(batch):
    """
    special collate_fn function for UDF class TrainDatasetTriplet
    :param batch: 
    :return: 
    """
    # batch_size = len(batch)
    images = []
    labels = []

    for b in batch:
        if b[0] is None:
            continue
        else:
            images.extend(b[0])  # extend will transfer torch.[B,C,H,W]->list(torch.[C,H,W])
            labels.extend(b[1])

    images = torch.stack(images, 0)  # images : list of [C,H,W] -> [Len_of_list, C, H,W]
    labels = torch.from_numpy(np.array(labels))
    assert (images.shape[0] == labels.shape[0])
    return images, labels


# =======================================================================================================================
# Specific version of triplet loss data loader and data aug is done with batch. Also, image is added with background.
# =======================================================================================================================
class TrainDatasetTripletBatchAug_BG(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, debug: bool = True, name: str = 'train', imgsize=256,
                 class_num=-1):
        super().__init__()
        self._root = root
        self._df = df
        self._debug = debug
        self._name = name
        self._imgsize = imgsize
        self._class_num = class_num

    def __len__(self):  # how much times will each epoch sample
        return min(max(len(self._df), 20000), 15000)
        # return self._class_num*125

    @staticmethod
    def tbatch():
        return 4

    def __getitem__(self, idx: int):
        # choose label from data a
        # choose any tow sample from data b bcz 1 image per class
        labelA = int(idx % self._class_num)

        dfA = self._df[self._df['attribute_ids'] == labelA]
        while dfA.empty:
            labelA = random.randint(0, self._class_num - 1)
            dfA = self._df[self._df['attribute_ids'] == labelA]

        len_dfA = len(dfA)
        assert (len_dfA != 0)
        pair_idxA = [random.randint(0, len_dfA - 1) for _ in range(2)]  # 有重采样

        images = []
        targets = []

        # pos
        for idxA in pair_idxA:
            item = dfA.iloc[idxA]
            image = load_image(item, self._root)
            # real_ comes from matterport, val_ comes from FFE without rles, they are added into training @ this version
            if item.id.startswith('real') or item.id.startswith('val'):
                bg_Flag = False
            else:
                bg_Flag = True
            image = rand_bg_resize_crop(image, item.id, imgsize=(self._imgsize, self._imgsize), addBg=bg_Flag)
            lb = int(item.attribute_ids)
            assert (lb < self._class_num)
            images.append(image)
            targets.append(lb)

        # neg
        dfB = self._df[self._df['attribute_ids'] != labelA]
        len_dfB = len(dfB)
        pair_idxB = [random.randint(0, len_dfB - 1) for _ in range(2)]  # 有重采样

        for idxB in pair_idxB:
            item = dfB.iloc[idxB]
            image = load_image(item, self._root)
            if item.id.startswith('real') or item.id.startswith('val'):
                bg_Flag = False
            else:
                bg_Flag = True
            image = rand_bg_resize_crop(image, item.id, imgsize=(self._imgsize, self._imgsize), addBg=bg_Flag)
            images.append(image)
            lb = int(item.attribute_ids)
            targets.append(lb)

        # IAA Batch Operations
        # print(images[0].shape)
        images = np.stack(images, 0)

        # images = [B,H,W,C]
        images = iaa_transformer.act_batch(images)
        images = np.transpose(images, (0, 3, 1, 2))  # [B,H,W,C] -> [B,C,H,W]
        images = images.astype(np.float32)
        images = images / 255.0

        return torch.FloatTensor(images), targets


def collate_TrainDatasetTripletBatchAug_BG(batch):
    """
    special collate_fn function for UDF class TrainDatasetTriplet
    :param batch: 
    :return: 
    """
    # batch_size = len(batch)
    images = []
    labels = []

    for b in batch:
        if b[0] is None:
            continue
        else:
            images.extend(b[0])
            labels.extend(b[1])

    images = torch.stack(images, 0)  # images : list of [C,H,W] -> [Len_of_list, C, H,W]
    labels = torch.from_numpy(np.array(labels))
    assert (images.shape[0] == labels.shape[0])
    return images, labels


# =======================================================================================================================
# Specific version of data loader and data aug is done with batch. Also, image is added with background. Label is bbox
# [cx,cy,w,h] -> Real(0,1)
# =======================================================================================================================
class TrainDatasetBatchAug_BG_4_BBox(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, debug: bool = True, name: str = 'train', imgsize=256,
                 class_num=-1):
        super().__init__()
        self._root = root
        self._df = df
        self._debug = debug
        self._name = name
        self._imgsize = imgsize
        self._class_num = class_num

    def __len__(self):  # how much times will each epoch sample
        return min(max(len(self._df), 20000), 15000)
        # return self._class_num*125

    @staticmethod
    def tbatch():
        return 1

    def __getitem__(self, idx: int):
        # choose label from data a
        # choose any tow sample from data b bcz 1 image per class
        # images = []
        # targets = []

        idxA = int(idx % len(self._df))
        item = self._df.iloc[idxA]
        image = load_image(item, self._root)  # [H,W,C]
        image, bbox = rand_bg_resize_crop_withbbox(image, item.id, imgsize=(self._imgsize, self._imgsize))

        # save_img_debug(image, [0.5,0.5,1,1])

        bbox = np.array(bbox).reshape(1, 4)

        return image, bbox




        # images = [H,W,C]

        # image = iaa_transformer.act(image)
        # image = np.transpose(image, (2,0,1)) #[H,W,C] -> [C,H,W]
        # image = image.astype(np.float32)
        # image = image / 255.0

        # return torch.FloatTensor(image),torch.FloatTensor(bbox).reshape(1,4)


def collate_TrainDatasetBatchAug_BG_4_BBox(batch):
    """
    special collate_fn function for UDF class TrainDatasetTriplet
    :param batch:
    :return:
    """
    # batch_size = len(batch)
    images = []
    labels = []

    for b in batch:
        if b[0] is None:
            continue
        else:
            images.append(b[0])
            labels.append(b[1])

    images = np.stack(images, 0)  # [H,W,C] -> [B,H,W,C]

    images = iaa_transformer.act_batch(images)
    images = np.transpose(images, (0, 3, 1, 2))  # [B,H,W,C] -> [B,C,H,W]
    images = images.astype(np.float32)
    images = images / 255.0

    images = torch.FloatTensor(images)

    labels = np.concatenate(labels, axis=0)
    labels = torch.from_numpy(labels)
    labels = labels.float()
    assert (images.shape[0] == labels.shape[0])
    return images, labels


# =======================================================================================================================
# Specific version of data loader for loading data from 2 kind of database. One can only be used for creating N pairs.
# =======================================================================================================================
# #item \
# # - attribute_ids - id - folds - data: a,b
class TrainDatasetSelected(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, debug: bool = True, name: str = 'train', imgsize=256,
                 class_num=-1):
        super().__init__()
        self._root = root
        self._df = df
        self._debug = debug
        self._name = name
        self._imgsize = imgsize
        self._dfA = df[df['data'] == 'a']
        self._dfB = df[df['data'] == 'b']
        self._class_num = class_num

    def __len__(self):
        return len(self._df) // 4

    def __getitem__(self, idx: int):
        # choose label from data a
        # choose any tow sample from data b bcz 1 image per class
        labelA = int(idx % self._class_num)
        # https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas
        # dfA = self._df[(self._df['data'] == 'a')&(self._df['attribute_ids'] == str(labelA))]
        dfA = self._dfA[self._dfA['attribute_ids'] == labelA + 1]
        len_dfA = len(dfA)
        pair_idxA = [random.randint(0, len_dfA - 1) for _ in range(2)]

        imagesA = []
        imagesB = []
        single_targetsA = []
        single_targetsB = []

        for idxA in pair_idxA:
            item = dfA.iloc[idxA]
            image = load_transform_image(item, self._root, imgsize=self._imgsize, debug=self._debug, name=self._name)
            lb = int(item.attribute_ids) - 1
            assert (lb < self._class_num)
            imagesA.append(image)
            single_targetsA.append(lb)

        dfB = self._dfB
        len_dfB = len(dfB)
        pair_idxB = [random.randint(0, len_dfB - 1) for _ in range(2)]

        for idxB in pair_idxB:
            item = dfB.iloc[idxB]
            image = load_transform_image(item, self._root, imgsize=self._imgsize, debug=self._debug, name=self._name)
            imagesB.append(image)
            lb = int(item.attribute_ids) - 1
            single_targetsB.append(lb)

        return (imagesA, imagesB), (single_targetsA, single_targetsB)


def collate_TrainDatasetSelected(batch):
    """
    special collate_fn function for UDF class TrainDatasetSelected
    :param batch: 
    :return: 
    """
    # batch_size = len(batch)
    imagesA = []
    imagesB = []
    labelsA = []
    labelsB = []

    for b in batch:
        if b[0] is None:
            continue
        else:
            imagesA.extend(b[0][0])
            imagesB.extend(b[0][1])
            labelsA.extend(b[1][0])
            labelsB.extend(b[1][1])

    assert (len(imagesA) == len(imagesB))
    imagesA.extend(imagesB)
    labelsA.extend(labelsB)

    imagesA = torch.stack(imagesA, 0)  # images : list of [C,H,W] -> [Len_of_list, C, H,W]
    labelsA = torch.from_numpy(np.array(labelsA))
    return imagesA, labelsA


# =======================================================================================================================
# data loader function for company location score
# =======================================================================================================================
class TrainDatasetLocationRS(Dataset):
    def __init__(self, df_comp_feat: pd.DataFrame,
                 df_loc_feat: pd.DataFrame,
                 df_pair: pd.DataFrame,
                 df_ensemble_score, flag_ensemble: bool,
                 emb_dict: dict, citynum=5,
                 name: str = 'train', posN=100, negN=200, testStep=500000):
        super().__init__()
        self._df_comp_feat = df_comp_feat.fillna(0)
        self._df_loc_feat = df_loc_feat.fillna(0)
        self._df_pair = df_pair.reset_index()
        self._df_ensemble_score = df_ensemble_score.reset_index(drop=True)
        self._name = name
        self._posN = posN
        self._negN = negN
        self._step = testStep
        self._emb_dict = emb_dict
        self._flag_ensemble = flag_ensemble
        self._not_cols = ['duns_number', 'atlas_location_uuid', 'label', 'city']

    def __len__(self):
        if self._name == 'train':
            return 1000
        else:
            return math.ceil(len(self._df_pair) / self._step)  # len of pair

    def tbatch(self):
        return self._posN + self._negN

    def __getitem__(self, idx: int):
        if self._name == 'train':
            # sample a part of data from training pair as positive seed
            dat1 = self._df_pair.sample(n=self._posN).reset_index(drop=True)
            dat2 = dat1.sample(frac=1).reset_index(drop=True)

            # generate negative sample from positive seed
            twin_dat = pd.merge(dat1, dat2, on='city', how='left', suffixes=['_left', '_right'])
            # twin_dat = twin_dat[twin_dat['atlas_location_uuid_left'] != twin_dat['atlas_location_uuid_right']]
            pot_neg_datA = twin_dat[
                ['duns_number_left', 'atlas_location_uuid_right']] \
                .rename(columns={'duns_number_left': 'duns_number', 'atlas_location_uuid_right': 'atlas_location_uuid'})

            pot_neg_datB = twin_dat[
                ['duns_number_right', 'atlas_location_uuid_left']] \
                .rename(columns={'duns_number_right': 'duns_number', 'atlas_location_uuid_left': 'atlas_location_uuid'})

            pot_neg_dat = pd.concat([pot_neg_datA, pot_neg_datB], axis=0)
            pot_neg_dat['label'] = 0
            dat1['label'] = 1

            # col alignment
            col_list = ['duns_number', 'atlas_location_uuid', 'label']
            dat1 = dat1[col_list]
            pot_neg_dat = pot_neg_dat[col_list]

            # clean pos dat in neg dat
            neg_dat = pd.merge(pot_neg_dat, dat1, on=['duns_number', 'atlas_location_uuid'], how='left',
                               suffixes=['', '_right']).reset_index(drop=True)
            neg_dat['label'] = neg_dat['label'].fillna(0)
            neg_dat = neg_dat[neg_dat['label_right'] != 1]

            neg_dat = neg_dat[['duns_number', 'atlas_location_uuid', 'label']].sample(
                n=min(self._negN, len(neg_dat))).reset_index(drop=True)

            pos_dat = dat1[col_list]
            res_dat = pd.concat([pos_dat, neg_dat], axis=0)
            res_dat = res_dat.sample(frac=1).reset_index(drop=True)
            Label = res_dat[['label']].to_numpy()
        else:
            inds = idx * self._step
            inde = min((idx + 1) * self._step, len(self._df_pair)) - 1  # loc[a,b] = [a,b] close set!!
            # res_dat = self._df_pair.loc[inds:inde,['duns_number','atlas_location_uuid','groundtruth']]
            # Label = (res_dat['atlas_location_uuid'] == res_dat['groundtruth']).to_numpy() + 0
            res_dat = self._df_pair.loc[inds:inde, ['duns_number', 'atlas_location_uuid', 'label']]
            Label = res_dat[['label']].to_numpy()

        # concate training pair with location/company feature
        F_res_dat = pd.merge(res_dat, self._df_comp_feat, on='duns_number', how='left')
        list_col = list(self._df_comp_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        FeatComp = F_res_dat[list_col].to_numpy()

        F_res_dat = pd.merge(res_dat, self._df_loc_feat, on='atlas_location_uuid', how='left')
        list_col = list(self._df_loc_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        # print(list_col)
        FeatLoc = F_res_dat[list_col].to_numpy()

        if self._flag_ensemble:
            F_res_dat = pd.merge(res_dat, self._df_ensemble_score, on=['atlas_location_uuid', 'duns_number'],
                                 how='left')
            list_col = list(self._df_ensemble_score.columns)
            list_col = [col for col in list_col if col not in self._not_cols]
            # print(list_col)
            FeatEnsembleScore = F_res_dat[list_col].to_numpy()
        else:
            FeatEnsembleScore = np.ones((len(F_res_dat), 1), dtype=np.float32)

        # trans id(str) 2 Long
        loc_name_str = res_dat['atlas_location_uuid'].values.tolist()
        loc_name_int = [self._emb_dict[n] for n in loc_name_str]

        # [B,Len_feat],[B,1]
        assert (len(Label) == len(FeatComp) and len(Label) == len(FeatLoc))
        # print(Label.sum(), FeatLoc.sum(),FeatComp.sum())

        featComp = torch.FloatTensor(FeatComp)
        featLoc = torch.FloatTensor(FeatLoc)
        featEnsembleScore = torch.FloatTensor(FeatEnsembleScore)
        featId = torch.LongTensor(loc_name_int).reshape(-1, 1)
        target = torch.LongTensor(Label).reshape(-1, 1)

        return {"feat_comp": featComp,
                "feat_loc": featLoc,
                "target": target,
                "feat_id": featId,
                "feat_ensemble_score": featEnsembleScore,
                "feat_comp_dim": FeatComp.shape,
                "feat_loc_dim": FeatLoc.shape}


def collate_TrainDatasetLocationRS(batch):
    """
    special collate_fn function for UDF class TrainDatasetTriplet
    :param batch: 
    :return: 
    """
    feat_comp = []
    feat_loc = []
    feat_id = []
    feat_ensemble_score = []
    labels = []

    for b in batch:
        feat_comp.append(b['feat_comp'])
        feat_loc.append(b['feat_loc'])
        feat_id.append(b['feat_id'])
        feat_ensemble_score.append(b['feat_ensemble_score'])
        labels.append(b['target'])

    feat_comp = torch.cat(feat_comp, 0)
    feat_loc = torch.cat(feat_loc, 0)
    feat_id = torch.cat(feat_id, 0)
    feat_ensemble_score = torch.cat(feat_ensemble_score, 0)
    labels = torch.cat(labels, 0)
    # print(feat_comp.shape,feat_loc.shape,labels.shape)

    assert (feat_loc.shape[0] == labels.shape[0])
    assert (feat_comp.shape[0] == labels.shape[0])
    assert (feat_id.shape[0] == labels.shape[0])
    assert (feat_ensemble_score.shape[0] == labels.shape[0])
    return {
        "feat_comp": feat_comp,
        "feat_loc": feat_loc,
        "feat_id": feat_id,
        "feat_ensemble_score": feat_ensemble_score,
        "target": labels
    }


# =======================================================================================================================
# data loader function for company location region modelling
# RSRB: Recommendation System Region Based
# =======================================================================================================================
class TrainDatasetLocationRSRB(Dataset):
    def __init__(self, df_comp_feat: pd.DataFrame,
                 df_loc_feat: pd.DataFrame,
                 df_pair: pd.DataFrame,
                 citynum=5,
                 name: str = 'train', trainStep=10000, testStep=500000):
        super().__init__()
        self._df_comp_feat = df_comp_feat.fillna(0)
        self._df_loc_feat = df_loc_feat.fillna(0)
        self._df_pair = df_pair.reset_index()
        self._name = name
        self._step = testStep
        self._citynum = citynum
        self._maxK = 50
        self._traintimes = trainStep
        self.cldat = []
        self.locname = []
        self.df_comp_feat_city = []
        if name in ['train', 'train_fast']:
            for ind_city in range(citynum):
                self.cldat.append(self._df_pair[(self._df_pair['fold'] == 0) & (self._df_pair['city'] == ind_city)])
                self.locname.append(self.cldat[ind_city].groupby('atlas_location_uuid').head(1).reset_index(drop=True)[
                                        ['atlas_location_uuid']])
                self.df_comp_feat_city.append(
                    self._df_comp_feat[self._df_comp_feat['city'] == ind_city].reset_index(drop=True))
        self._debug = False
        self._not_cols = ['duns_number', 'atlas_location_uuid', 'label', 'city']

    def __len__(self):
        if self._name in ['train', 'train_fast']:
            return self._traintimes
        else:
            return math.ceil(len(self._df_pair) / self._step)  # len of pair

    def tbatch(self):
        return 0

    def __getitem__(self, idx: int):
        tc = timer(display=self._debug)
        if self._name == 'train':
            pass
            # #pick a city randomly
            # ind_city = math.floor(random.random() * self._citynum)
            # cldat = self.cldat[ind_city]
            #
            # fn = lambda obj: obj.loc[np.random.choice(obj.index, 1, True), :]
            # cldatGrp = cldat.groupby('atlas_location_uuid')
            # tbA = cldatGrp.apply(fn).reset_index(drop=True)[
            #     ['duns_number', 'atlas_location_uuid']]
            # # print('1.len of tbA %d:' % len(tbA))
            # fn = lambda obj: obj.loc[np.random.choice(obj.index, self._maxK, True), :]
            # tbB = cldatGrp.apply(fn).reset_index(drop=True)[
            #     ['duns_number', 'atlas_location_uuid']]
            # # print('1.len of tbB %d' % len(tbB))
            #
            # ###======================Pos=============================###
            # tbA['mk'] = 'A'
            # tbB = tbB.merge(tbA, on=['duns_number', 'atlas_location_uuid'], how='left', suffixes=['', '_right'])
            # tbB = tbB[tbB['mk'].isnull()]
            # # print('2.len of tbB not included in tbA %d' % len(tbB))
            # # we need to full fill the data
            # tbBGrp = tbB.groupby('atlas_location_uuid')
            # tbB = tbBGrp.apply(fn).reset_index(drop=True)[
            #     ['duns_number', 'atlas_location_uuid']]
            # tbB['mk'] = 'B'
            # # print('3.len of tbB full filled again %d' % len(tbB))
            # # in case tbB cut some locations from tbA, lets shrink tbA
            # tblocB = tbBGrp.first().reset_index()
            # tblocB['mk'] = 'B'
            # # print('4.len of locations in tbB %d' % len(tblocB))
            # tbA = tbA.merge(tblocB, on='atlas_location_uuid', how='left', suffixes=['', '_right'])
            # tbA = tbA[tbA['mk_right'].notnull()][['duns_number', 'atlas_location_uuid', 'mk']].reset_index(drop=True)
            # # print('4.len of tbA with common locations of tbB %d' % len(tbA))
            #
            # ###======================Neg=============================###
            # tbAA = pd.concat([tbA, tbA.sample(frac=1).reset_index() \
            #                  .rename(
            #     columns={'duns_number': 'duns_number_n', 'atlas_location_uuid': 'atlas_location_uuid_n', 'mk': 'mk_n'})]
            #                  , axis=1)
            # # print('5.len of negpair %d' % len(tbAA))
            # tbAA = tbAA.merge(cldat, \
            #                   left_on=['duns_number_n', 'atlas_location_uuid'],
            #                   right_on=['duns_number', 'atlas_location_uuid'], \
            #                   how='left', suffixes=['', '_right'])
            #
            # tbC = tbAA[tbAA['duns_number_right'].isnull()][['duns_number_n', 'atlas_location_uuid']] \
            #     .rename(columns={'duns_number_n': 'duns_number'})
            # # print('6.len of neg data %d' % len(tbC))
            #
            # # in case tbC cut some locations from tbA and tbB
            # tbC['mk'] = 'C'
            # tblocC = tbC.groupby('atlas_location_uuid').first().reset_index()
            # # print('6.locations in neg data %d' % len(tblocC))
            # tbA = tbA.merge(tblocC, on='atlas_location_uuid', how='left', suffixes=['', '_right'])
            # tbA = tbA[tbA['mk_right'].notnull()][['duns_number', 'atlas_location_uuid', 'mk']].reset_index(drop=True)
            # # print('final tbA len %d' % len(tbA))
            #
            # tbB = tbB.merge(tblocC, on='atlas_location_uuid', how='left', suffixes=['', '_right'])
            # tbB = tbB[tbB['mk_right'].notnull()][['duns_number', 'atlas_location_uuid', 'mk']].reset_index(drop=True)
            # # print('final tbB len %d' % len(tbB))
            #
            # tbA = tbA.sort_values(by='atlas_location_uuid')
            # tbB = tbB.sort_values(by='atlas_location_uuid')
            # tbC = tbC.sort_values(by='atlas_location_uuid')
            #
            # assert (len(tbA) == len(tbC) and len(tbB) == len(tbA) * self._maxK)
            #
            # list_col = list(self._df_comp_feat.columns)
            # list_col = [col for col in list_col if col not in ['duns_number', 'atlas_location_uuid', 'label','city']]
            #
            # featA = tbA.merge(self._df_comp_feat,on='duns_number',how='left',suffixes=['','_right'])[list_col]
            # featB = tbB.merge(self._df_comp_feat,on='duns_number',how='left',suffixes=['','_right'])[list_col]
            # featC = tbC.merge(self._df_comp_feat,on='duns_number',how='left',suffixes=['','_right'])[list_col]
        elif self._name == 'train_fast':
            num_building_batch = 20
            num_pos = 50  # each building
            num_region = self._maxK

            data_batch = num_pos + num_pos * num_region
            num_pos_pair = num_pos * num_building_batch
            # num_neg_pair = 2*num_pos_pair

            ind_city = math.floor(random.random() * self._citynum)
            cldat = self.cldat[ind_city]

            tc.start(it='location selection')
            smp_loc_name = self.locname[ind_city].sample(n=num_building_batch).reset_index(drop=True)
            smp_cldat = cldat.merge(smp_loc_name, on='atlas_location_uuid', how='inner', suffixes=['', '_right'])
            tc.eclapse()

            tc.start(it='sample pos and region data')
            cldatGrp = smp_cldat.groupby('atlas_location_uuid')
            tbAB = cldatGrp.apply(lambda x: x.sample(data_batch, replace=True)).reset_index(drop=True)[
                ['duns_number', 'atlas_location_uuid']]
            tc.eclapse()

            tc.start(it='create tbA and tbB')
            tbABGrp = tbAB.groupby('atlas_location_uuid')
            tbA = tbABGrp.head(num_pos).reset_index(drop=True)

            tbB = tbABGrp.tail(num_pos * num_region).reset_index(drop=True)
            tc.eclapse()

            assert (len(tbA) == num_pos_pair)

            tc.start(it='get location neg pairs')
            smp_loc_name_pair = \
                pd.concat([smp_loc_name,
                           smp_loc_name.sample(frac=1, replace=True).reset_index(drop=True) \
                          .rename(columns={'atlas_location_uuid': 'atlas_location_uuid_neg'})], axis=1)

            tbC = \
            smp_loc_name_pair.merge(tbA, left_on='atlas_location_uuid_neg', right_on='atlas_location_uuid', how='inner',
                                    suffixes=['', '_useless'])[
                ['duns_number', 'atlas_location_uuid', 'atlas_location_uuid_neg']].reset_index(drop=True)
            tc.eclapse()

            tc.start('sort')
            tbA = tbA.sort_values(by='atlas_location_uuid').reset_index(drop=True)
            tbB = tbB.sort_values(by='atlas_location_uuid').reset_index(drop=True)
            tbC = tbC.sort_values(by='atlas_location_uuid').reset_index(drop=True)
            tc.eclapse()

            list_col = list(self._df_comp_feat.columns)
            list_col = [col for col in list_col if col not in self._not_cols]

            tc.start('merge')
            tbACB = pd.concat([tbA, tbC, tbB], axis=0, sort=False).reset_index(drop=True)
            featACB_comp = \
            tbACB.merge(self.df_comp_feat_city[ind_city], on='duns_number', how='left', suffixes=['', '_right'])[
                list_col]

            featA = featACB_comp.loc[:num_pos_pair - 1]
            featC = featACB_comp.loc[num_pos_pair:2 * num_pos_pair - 1]
            featB = featACB_comp.loc[2 * num_pos_pair:]

            assert (len(featC) == len(featA))

            list_col = list(self._df_loc_feat.columns)
            list_col = [col for col in list_col if col not in self._not_cols]
            # tbA and tbB share the same location, thus tbA is used.
            featB_loc = tbA.merge(self._df_loc_feat, on='atlas_location_uuid', how='left', suffixes=['', '_right'])[
                list_col]
            tc.eclapse()

        else:
            dataLen = len(self._df_pair[self._df_pair['mk'] == 'A'])
            dataLenB = len(self._df_pair[self._df_pair['mk'] == 'B'])
            inds = idx * self._step
            inde = min((idx + 1) * self._step, dataLen) - 1  # loc[a,b] = [a,b] close set!!
            # res_dat = self._df_pair.loc[inds:inde, ['duns_number', 'atlas_location_uuid','city','mk']]

            indsB = idx * self._step * self._maxK
            indeB = min((idx + 1) * self._step * self._maxK, dataLenB) - 1  # loc[a,b] = [a,b] close set!!

            list_col = list(self._df_comp_feat.columns)
            list_col = [col for col in list_col if col not in self._not_cols]

            datA = self._df_pair[self._df_pair['mk'] == 'A'].sort_values(
                by=['city', 'atlas_location_uuid']).reset_index(drop=True)
            datB = self._df_pair[self._df_pair['mk'] == 'B'].sort_values(
                by=['city', 'atlas_location_uuid']).reset_index(drop=True)
            datC = self._df_pair[self._df_pair['mk'] == 'C'].sort_values(
                by=['city', 'atlas_location_uuid']).reset_index(drop=True)

            datA = datA.loc[inds:inde, ['duns_number', 'atlas_location_uuid', 'city', 'mk']]
            datB = datB.loc[indsB:indeB, ['duns_number', 'atlas_location_uuid', 'city', 'mk']]
            datC = datC.loc[inds:inde, ['duns_number', 'atlas_location_uuid', 'city', 'mk']]

            featA = datA.merge(self._df_comp_feat, on='duns_number', how='left', suffixes=['', '_right'])[list_col]
            featB = datB.merge(self._df_comp_feat, on='duns_number', how='left', suffixes=['', '_right'])[list_col]
            featC = datC.merge(self._df_comp_feat, on='duns_number', how='left', suffixes=['', '_right'])[list_col]

            list_col = list(self._df_loc_feat.columns)
            list_col = [col for col in list_col if col not in self._not_cols]
            featB_loc = datA.merge(self._df_loc_feat, on='atlas_location_uuid', how='left', suffixes=['', '_right'])[
                list_col]

        # all branch need such operation...
        tc.start('Transfer storage')
        featA, featB, featC, featB_loc = featA.to_numpy(), featB.to_numpy(), featC.to_numpy(), featB_loc.to_numpy()

        featCompPos = torch.FloatTensor(featA)  # B,D
        featRegion = torch.FloatTensor(featB)
        featLoc = torch.FloatTensor(featB_loc)
        N, featdim = featRegion.shape
        # print(featA.shape,featB.shape,featC.shape)
        assert (N == featCompPos.shape[0] * self._maxK)

        featRegion = featRegion.view(-1, self._maxK, featdim)  # B,K,D

        featCompNeg = torch.FloatTensor(featC)  # B,D
        tc.eclapse()

        return {
            "feat_comp_pos": featCompPos,
            "feat_comp_neg": featCompNeg,
            "feat_comp_region": featRegion,
            "feat_loc": featLoc,
        }


def collate_TrainDatasetLocationRSRB(batch):
    """
    special collate_fn function for UDF class TrainDatasetTriplet
    :param batch: 
    :return: 
    """
    feat_comp_pos = []
    feat_comp_neg = []
    feat_comp_region = []
    feat_loc = []

    for b in batch:
        feat_comp_pos.append(b['feat_comp_pos'])
        feat_comp_neg.append(b['feat_comp_neg'])
        feat_comp_region.append(b['feat_comp_region'])
        feat_loc.append(b['feat_loc'])

    feat_comp_pos = torch.cat(feat_comp_pos, 0)
    feat_comp_neg = torch.cat(feat_comp_neg, 0)
    feat_comp_region = torch.cat(feat_comp_region, 0)
    feat_loc = torch.cat(feat_loc, 0)
    # print(feat_comp.shape,feat_loc.shape,labels.shape)

    assert (feat_comp_pos.shape[0] == feat_comp_neg.shape[0] and feat_comp_region.shape[0] == feat_comp_pos.shape[0] and
            feat_loc.shape[0] == feat_comp_pos.shape[0])

    return {
        "feat_comp_pos": feat_comp_pos,
        "feat_comp_neg": feat_comp_neg,
        "feat_comp_region": feat_comp_region,
        "feat_loc": feat_loc,
    }

class TestDatasetLocationRSRB(Dataset):
    def __init__(self, df_comp_feat: pd.DataFrame,
                 df_loc_feat: pd.DataFrame,
                 df_region_feat:pd.DataFrame,
                 df_pair: pd.DataFrame,
                 citynum=5, testStep=500000):
        super().__init__()
        self._df_comp_feat = df_comp_feat.fillna(0)
        self._df_loc_feat = df_loc_feat.fillna(0)
        self._df_region_feat = df_region_feat.fillna(0)
        self._df_pair = df_pair.reset_index()
        self._step = testStep
        self._citynum = citynum
        self.cldat = []
        self.locname = []

        self._debug = False
        self._not_cols = ['duns_number', 'atlas_location_uuid', 'label', 'city']

    def __len__(self):
        return math.ceil(len(self._df_pair) / self._step)  # len of pair

    def tbatch(self):
        return 0

    def __getitem__(self, idx: int):
        tc = timer(display=self._debug)
        dataLen = len(self._df_pair)
        inds = idx * self._step
        inde = min((idx + 1) * self._step, dataLen) - 1

        datA = self._df_pair.loc[inds:inde, ['duns_number', 'atlas_location_uuid','label']]

        tc.start('Append feature with pairs')
        list_col = list(self._df_comp_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        featA_comp = datA.merge(self._df_comp_feat, on='duns_number', how='left', suffixes=sfx)[list_col]

        list_col = list(self._df_region_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        featA_region = datA.merge(self._df_region_feat, on='atlas_location_uuid', how='left', suffixes=sfx)[list_col]


        list_col = list(self._df_loc_feat.columns)
        list_col = [col for col in list_col if col not in self._not_cols]
        featA_loc = datA.merge(self._df_loc_feat, on='atlas_location_uuid', how='left', suffixes=sfx)[
            list_col]
        tc.eclapse()

        # all branch need such operation...
        tc.start('Transfer storage')
        featA_comp, featA_region, featA_loc = featA_comp.to_numpy(), featA_region.to_numpy(), featA_loc.to_numpy()

        featComp = torch.FloatTensor(featA_comp)  # B,D
        featRegion = torch.FloatTensor(featA_region)
        featLoc = torch.FloatTensor(featA_loc)

        N, featdim = featRegion.shape

        assert ( (N == featComp.shape[0]) and (N==featLoc.shape[0]) )
        tc.eclapse()

        return {
            "feat_comp": featComp,
            "feat_region": featRegion,
            "feat_loc": featLoc,
        }


def collate_TestDatasetLocationRSRB(batch):
    """
    special collate_fn function for UDF class TrainDatasetTriplet
    :param batch: 
    :return: 
    """
    feat_comp = []
    feat_region = []
    feat_loc = []

    for b in batch:
        feat_comp.append(b['feat_comp'])
        feat_region.append(b['feat_comp_region'])
        feat_loc.append(b['feat_loc'])

    feat_comp = torch.cat(feat_comp, 0)
    feat_region = torch.cat(feat_region, 0)
    feat_loc = torch.cat(feat_loc, 0)
    # print(feat_comp.shape,feat_loc.shape,labels.shape)

    assert (feat_comp.shape[0] == feat_region.shape[0]  and
            feat_loc.shape[0] == feat_comp.shape[0])

    return {
        "feat_comp": feat_comp,
        "feat_region": feat_region,
        "feat_loc": feat_loc,
    }

# =======================================================================================================================

class TTADataset:
    def __init__(self, root: Path, df: pd.DataFrame, tta_code, imgsize=256):
        self._root = root
        self._df = df
        self._tta_code = tta_code
        self._imgsize = imgsize

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        item = self._df.iloc[idx % len(self._df)]
        image = load_test_image(item, self._root, self._tta_code, self._imgsize)
        return image, item.id


def load_transform_image(item, root: Path, imgsize=256, debug: bool = False, name: str = 'train'):
    image = load_image(item, root)

    if name == 'train':
        alpha = random.uniform(0, 0.2)
        image = do_brightness_shift(image, alpha=alpha)
        image = random_flip(image, p=0.5)
        angle = random.uniform(0, 1) * 360
        image = rotate(image, angle, center=None, scale=1.0)
        # ratio = random.uniform(0.75, 0.99)
        # image = random_cropping(image, ratio = ratio, is_random = True)
        # image = random_erasing(image, probability=0.5, sl=0.02, sh=0.4, r1=0.3)
    else:
        pass
        # image = random_cropping(image, ratio=0.85, is_random=False)

    # Padding
    # maintain ratio of length and height
    # imgH, imgW, nCh = image.shape
    # nimgW, nimgH = max(imgW, imgH), max(imgW, imgH)
    # offset_W = (nimgW - imgW) // 2
    # offset_H = (nimgH - imgH) // 2
    #
    # nimage = np.zeros((nimgH, nimgW, nCh), dtype=np.uint8)
    # nimage[offset_H:imgH+offset_H, offset_W:imgW+offset_W, :] = 1*image[:,:,:]

    # Crop
    if name == 'train':
        ratio = random.uniform(0.70, 0.99)
        nimage = random_cropping(image, ratio=ratio, is_random=True)
    else:
        nimage = random_cropping(image, ratio=0.8, is_random=False)

    # Resize
    image = cv2.resize(nimage, (imgsize, imgsize))

    if debug:
        image.save('_debug.png')

    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = image.reshape([-1, imgsize, imgsize])
    image = image / 255.0

    # is_venn = True
    # if is_venn:
    #     # mean = [0.485, 0.456, 0.406]
    #     # std = [0.229, 0.224, 0.225]
    #     image[0,:,:] = (image[0,:,:] - 0.485) / 0.229
    #     image[1,:,:] = (image[1,:,:] - 0.456) / 0.224
    #     image[2,:,:] = (image[2,:,:] - 0.406) / 0.225

    return torch.FloatTensor(image)


def load_test_image(item, root: Path, tta_code, imgsize):
    image = load_image(item, root)
    image = aug_image(image, augment=tta_code)
    image = cv2.resize(image, (imgsize, imgsize))

    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = image.reshape([-1, imgsize, imgsize])
    image = image / 255.0

    # is_venn = True
    # if is_venn:
    #     # mean = [0.485, 0.456, 0.406]
    #     # std = [0.229, 0.224, 0.225]
    #     image[0,:,:] = (image[0,:,:] - 0.485) / 0.229
    #     image[1,:,:] = (image[1,:,:] - 0.456) / 0.224
    #     image[2,:,:] = (image[2,:,:] - 0.406) / 0.225

    return torch.FloatTensor(image)


def load_image(item, root: Path) -> Image.Image:
    # print(str(root + '/' + f'{item.id}.jpg'))
    # image = cv2.imread(str(root + '/' + f'{item.id}'))
    image = cv2.imread(os.path.join(root, str(f'{item.id}')))
    # print(os.path.join(root,str(f'{item.id}')))
    # print()
    # print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# image name looks like : idx_copy.jpg
def get_ids(root: Path) -> List[str]:
    return sorted({p.name.split('_')[0] for p in root.glob('*.jpg')})


def load_transform_image_iaa(item, root: Path, imgsize=256, debug: bool = False, name: str = 'train'):
    image = load_image(item, root)

    if name == 'train':
        alpha = random.uniform(0, 0.2)
        image = do_brightness_shift(image, alpha=alpha)
        image = random_flip(image, p=0.5)
        angle = random.uniform(0, 1) * 360
        image = rotate(image, angle, center=None, scale=1.0)
        image_aug = iaa_transformer.act(image)
    else:
        image_aug = random_cropping(image, ratio=0.95, is_random=False)

    image = cv2.resize(image_aug, (imgsize, imgsize))

    if debug:
        image.save('_debug.png')

    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = image.reshape([-1, imgsize, imgsize])
    image = image / 255.0

    # is_venn = True
    # if is_venn:
    #     # mean = [0.485, 0.456, 0.406]
    #     # std = [0.229, 0.224, 0.225]
    #     image[0,:,:] = (image[0,:,:] - 0.485) / 0.229
    #     image[1,:,:] = (image[1,:,:] - 0.456) / 0.224
    #     image[2,:,:] = (image[2,:,:] - 0.406) / 0.225

    return torch.FloatTensor(image)


def load_image_uint8(item, root: Path, imgsize=256, debug: bool = False, name: str = 'train'):
    image = load_image(item, root)

    if name == 'train':
        alpha = random.uniform(0, 0.2)
        image = do_brightness_shift(image, alpha=alpha)
        image = random_flip(image, p=0.5)
        angle = random.uniform(0, 1) * 360
        image_aug = rotate(image, angle, center=None, scale=1.0)
    else:
        image_aug = random_cropping(image, ratio=0.8, is_random=False)

    image = cv2.resize(image_aug, (imgsize, imgsize))

    if debug:
        image.save('_debug.png')
    return image


def save_img_debug(img, bbox):
    h, w, c = img.shape
    bbox_abs = [bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h]
    bbox_abs = [bbox_abs[0] - bbox_abs[2] / 2, bbox_abs[1] - bbox_abs[3] / 2, bbox_abs[2], bbox_abs[3]]
    pt1 = [int(bbox_abs[0]), int(bbox_abs[1])]
    pt2 = [int(bbox_abs[0] + bbox_abs[2]), int(bbox_abs[1] + bbox_abs[3])]
    nimg = np.array(img)
    cv2.rectangle(nimg, tuple(pt1), tuple(pt2), (255, 0, 0), 2)

    postname = str(random.randint(0, 100))
    save_name = './img_tmp/m_debug_' + postname + '.jpeg'
    cv2.imwrite(save_name, nimg)
