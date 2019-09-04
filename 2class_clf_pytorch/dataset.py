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


# image_size = 256

iaa_transformer = iaaTransform()
iaa_transformer.getSeq()

class TrainDataset(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, debug: bool = True, name: str = 'train', imgsize = 256 , class_num = -1):
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
        image = load_transform_image_iaa(item, self._root, imgsize = self._imgsize,debug=self._debug, name=self._name)
        # target = torch.zeros(N_CLASSES)
        lb = item.attribute_ids
        # print(lb)

        # for cls in range(N_CLASSES):
        #     target[cls] = int(lb[cls + 1])
        # clsval = int(lb[5])
        # target = torch.from_numpy(np.array(item.attribute_ids))
        clsval = int(lb)
        assert(clsval>=0 and clsval < self._class_num)
        target = torch.from_numpy(np.array(clsval))
        return image, target


class TrainDatasetTriplet(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, debug: bool = True, name: str = 'train', imgsize=256, class_num = -1):
        super().__init__()
        self._root = root
        self._df = df
        self._debug = debug
        self._name = name
        self._imgsize = imgsize
        self._class_num = class_num

    def __len__(self):#how much times will each epoch sample
        return min(max(len(self._df),20000),40000)
        # return self._class_num*125

    @staticmethod
    def tbatch():
        return 8

    def __getitem__(self, idx: int):
        # choose label from data a
        # choose any tow sample from data b bcz 1 image per class
        labelA = int(idx % self._class_num)
        # print(labelA)
        # print('ds1')
        dfA = self._df[self._df['attribute_ids'] == labelA]
        while dfA.empty:
            labelA = random.randint(0,self._class_num-1)
            dfA = self._df[self._df['attribute_ids'] == labelA]

        # print('ds2')
        len_dfA = len(dfA)
        assert(len_dfA!=0)
        pair_idxA = [random.randint(0, len_dfA - 1) for _ in range(4)]#有重采样
        # print('ds3')
        images = []
        targets = []

        #pos
        for idxA in pair_idxA:
            # print('dsx')
            item = dfA.iloc[idxA]
            image = load_transform_image_iaa(item, self._root, imgsize=self._imgsize, debug=self._debug,
                                         name=self._name)
            # print('load done')
            lb = int(item.attribute_ids)
            assert (lb < self._class_num)
            images.append(image)
            targets.append(lb)

        #neg
        dfB = self._df[self._df['attribute_ids'] != labelA]
        len_dfB = len(dfB)
        pair_idxB = [random.randint(0, len_dfB - 1) for _ in range(4)]#有重采样

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

class TrainDatasetTripletBatchAug(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, debug: bool = True, name: str = 'train', imgsize=256, class_num = -1):
        super().__init__()
        self._root = root
        self._df = df
        self._debug = debug
        self._name = name
        self._imgsize = imgsize
        self._class_num = class_num

    def __len__(self):#how much times will each epoch sample
        return min(max(len(self._df),20000),40000)
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
            labelA = random.randint(0,self._class_num-1)
            dfA = self._df[self._df['attribute_ids'] == labelA]

        len_dfA = len(dfA)
        assert(len_dfA!=0)
        pair_idxA = [random.randint(0, len_dfA - 1) for _ in range(4)]#有重采样

        images = []
        targets = []

        #pos
        for idxA in pair_idxA:
            # print('dsx')
            item = dfA.iloc[idxA]
            image = load_image_uint8(item, self._root, imgsize=self._imgsize, debug=self._debug,
                                         name=self._name)
            # print('load done')
            lb = int(item.attribute_ids)
            assert (lb < self._class_num)
            images.append(image)
            targets.append(lb)

        #neg
        dfB = self._df[self._df['attribute_ids'] != labelA]
        len_dfB = len(dfB)
        pair_idxB = [random.randint(0, len_dfB - 1) for _ in range(4)]#有重采样

        for idxB in pair_idxB:
            item = dfB.iloc[idxB]
            image = load_image_uint8(item, self._root, imgsize=self._imgsize, debug=self._debug,
                                         name=self._name)
            images.append(image)
            lb = int(item.attribute_ids)
            targets.append(lb)


        #IAA Batch Operations
        images = np.stack(images,0)

        #images = [B,H,W,C]
        images = iaa_transformer.act_batch(images)
        images = np.transpose(images, (0, 3, 1, 2)) #[B,H,W,C] -> [B,C,H,W]
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
            images.extend(b[0])
            labels.extend(b[1])

    images = torch.stack(images, 0)  # images : list of [C,H,W] -> [Len_of_list, C, H,W]
    labels = torch.from_numpy(np.array(labels))
    assert (images.shape[0] == labels.shape[0])
    return images, labels

# #item \
# # - attribute_ids - id - folds - data: a,b
class TrainDatasetSelected(Dataset):
    def __init__(self, root: Path, df: pd.DataFrame, debug: bool = True, name: str = 'train', imgsize = 256 , class_num = -1):
        super().__init__()
        self._root = root
        self._df = df
        self._debug = debug
        self._name = name
        self._imgsize = imgsize
        self._dfA = df[df['data']=='a']
        self._dfB = df[df['data']=='b']
        self._class_num = class_num

    def __len__(self):
        return len(self._df)//4

    def __getitem__(self, idx: int):
        #choose label from data a
        #choose any tow sample from data b bcz 1 image per class
        labelA = int(idx % self._class_num)
        #https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas
        # dfA = self._df[(self._df['data'] == 'a')&(self._df['attribute_ids'] == str(labelA))]
        dfA = self._dfA[self._dfA['attribute_ids'] == labelA+1]
        len_dfA = len(dfA)
        pair_idxA = [random.randint(0, len_dfA-1) for _ in range(2)]

        imagesA = []
        imagesB = []
        single_targetsA = []
        single_targetsB = []

        for idxA in pair_idxA:
            item = dfA.iloc[idxA]
            image = load_transform_image(item, self._root, imgsize=self._imgsize, debug=self._debug, name=self._name)
            lb = int(item.attribute_ids) - 1
            assert(lb < self._class_num)
            imagesA.append(image)
            single_targetsA.append(lb)

        dfB = self._dfB
        len_dfB = len(dfB)
        pair_idxB = [random.randint(0, len_dfB-1) for _ in range(2)]

        for idxB in pair_idxB:
            item = dfB.iloc[idxB]
            image = load_transform_image(item, self._root, imgsize=self._imgsize, debug=self._debug, name=self._name)
            imagesB.append(image)
            lb = int(item.attribute_ids) - 1
            single_targetsB.append(lb)

        return (imagesA,imagesB), (single_targetsA,single_targetsB)


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

    imagesA = torch.stack(imagesA,0) #images : list of [C,H,W] -> [Len_of_list, C, H,W]
    labelsA = torch.from_numpy(np.array(labelsA))
    return imagesA,labelsA


class TTADataset:
    def __init__(self, root: Path, df: pd.DataFrame, tta_code , imgsize = 256):
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


def load_transform_image(item, root: Path, imgsize=256,debug: bool = False, name: str = 'train'):
    image = load_image(item, root)

    if name == 'train':
        alpha = random.uniform(0, 0.2)
        image = do_brightness_shift(image, alpha=alpha)
        image = random_flip(image, p=0.5)
        angle = random.uniform(0, 1)*360
        image = rotate(image, angle, center=None, scale=1.0)
        # ratio = random.uniform(0.75, 0.99)
        # image = random_cropping(image, ratio = ratio, is_random = True)
        #image = random_erasing(image, probability=0.5, sl=0.02, sh=0.4, r1=0.3)
    else:
        pass
        # image = random_cropping(image, ratio=0.85, is_random=False)

    #Padding
    # maintain ratio of length and height
    # imgH, imgW, nCh = image.shape
    # nimgW, nimgH = max(imgW, imgH), max(imgW, imgH)
    # offset_W = (nimgW - imgW) // 2
    # offset_H = (nimgH - imgH) // 2
    #
    # nimage = np.zeros((nimgH, nimgW, nCh), dtype=np.uint8)
    # nimage[offset_H:imgH+offset_H, offset_W:imgW+offset_W, :] = 1*image[:,:,:]

    #Crop
    if name == 'train':
        ratio = random.uniform(0.70, 0.99)
        nimage = random_cropping(image, ratio=ratio, is_random=True)
    else:
        nimage = random_cropping(image, ratio=0.8, is_random=False)

    #Resize
    image = cv2.resize(nimage ,(imgsize, imgsize))

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
    image = aug_image(image, augment = tta_code)
    image = cv2.resize(image ,(imgsize, imgsize))

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
    image = cv2.imread(str(f'{item.id}'))
    # print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# image name looks like : idx_copy.jpg
def get_ids(root: Path) -> List[str]:
    return sorted({p.name.split('_')[0] for p in root.glob('*.jpg')})




def load_transform_image_iaa(item, root: Path, imgsize=256,debug: bool = False, name: str = 'train'):
    image = load_image(item, root)

    if name == 'train':
        alpha = random.uniform(0, 0.2)
        image = do_brightness_shift(image, alpha=alpha)
        image = random_flip(image, p=0.5)
        angle = random.uniform(0, 1)*360
        image = rotate(image, angle, center=None, scale=1.0)
        image_aug = iaa_transformer.act(image)
    else:
        image_aug = random_cropping(image, ratio=0.8, is_random=False)

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

def load_image_uint8(item, root: Path, imgsize=256,debug: bool = False, name: str = 'train'):
    image = load_image(item, root)

    if name == 'train':
        alpha = random.uniform(0, 0.2)
        image = do_brightness_shift(image, alpha=alpha)
        image = random_flip(image, p=0.5)
        angle = random.uniform(0, 1)*360
        image_aug = rotate(image, angle, center=None, scale=1.0)
    else:
        image_aug = random_cropping(image, ratio=0.8, is_random=False)

    image = cv2.resize(image_aug, (imgsize, imgsize))

    if debug:
        image.save('_debug.png')

    # image = np.transpose(image, (2, 0, 1))
    # image = image.astype(np.float32)
    # image = image.reshape([-1, imgsize, imgsize])
    # image = image / 255.0

    # is_venn = True
    # if is_venn:
    #     # mean = [0.485, 0.456, 0.406]
    #     # std = [0.229, 0.224, 0.225]
    #     image[0,:,:] = (image[0,:,:] - 0.485) / 0.229
    #     image[1,:,:] = (image[1,:,:] - 0.456) / 0.224
    #     image[2,:,:] = (image[2,:,:] - 0.406) / 0.225

    return image

