#Example of extracting feature using CPU where model is trained by GPU par
import models.models as models
from dataset import *
from utils import (write_event, load_par_gpu_model_cpu)
from pathlib import Path
import torch
from aug import *
from torch import nn, cuda
import cv2
from sklearn.metrics.pairwise import euclidean_distances
from shutil import copyfile



def torch_load_image(patch):
    image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    image = cropping(image)
    image = cv2.resize(image ,(256, 256))
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = image.reshape([-1, 256, 256])
    image = image / 255.0
    return torch.FloatTensor(image)


def cosDist(matA:np.ndarray,matB:np.ndarray):
    """
    :param matA: [trN,featdim]
    :param matB: [ttN,featdim]
    :return: [ttN,trN]
    """
    trN,featdim = matA.shape
    ttN,featdim2 = matB.shape
    assert(featdim2==featdim)
    BTA = matB @ np.transpose(matA)

    normA = matA ** 2
    normA = np.sum(normA, 1).reshape(1, -1)
    normA = np.sqrt(normA)
    normA = np.tile(normA, [ttN, 1])

    normB = matB ** 2
    normB = np.sum(normB, 1).reshape(-1, 1)
    normB = np.sqrt(normB)
    normB = np.tile(normB, [1, trN])

    normAB = normA * normB
    cosAB = BTA / normAB

    return cosAB








model_root = '/Users/yefeichen/Desktop/Work/Project/pyTorchClassification/2class_clf_pytorch/result/furniture_toy'
ckpt = 'model_loss_best.pt'
model_name = 'resnet50V4'

data_root = '/Users/yefeichen/Database/furniture/furniture_red_tech/'
# test_root = '/Users/yefeichen/Database/furniture/furniture_wework_real/'
save_root = '/Users/yefeichen/Database/furniture/furniture_red_tech_new/'
bad_root = '/Users/yefeichen/Database/furniture/bad/'
N_Cls = 128

model_root = Path(model_root)

print('weights loaded!')

fcnt = 0
ecnt = 0

#loading npy
featList = []
nameList = []
fileList = os.listdir(data_root)
for _file in fileList:
    if '.npy' in _file:
        npPath = os.path.join(data_root,_file)
        feat = np.load(npPath).reshape(1,-1)
        featList.append(feat)
        nameSp = os.path.splitext(_file)
        nameList.append(nameSp[0]+'.jpg')

trFeat = np.vstack(featList)

print('Features loaded from data: ' + str(trFeat.shape) )


# dist = cosDist(trFeat,trFeat)
pairDist = euclidean_distances(trFeat,trFeat)
rs,cs = pairDist.shape

ind_mat = np.ones((1,rs))

for r in range(rs):
    if ind_mat[0,r] == 1:
        dist = pairDist[r,:].reshape(1,-1)

        dist[0,0:r+1] = 10 # self distance
        ind_mat[dist < 0.01] = 0
    else:
        pass

print( str(rs - ind_mat.sum()) + ' images cut' )

for r in range(rs):
    if ind_mat[0,r] == 1:
        imgName = nameList[r]
        imgPath = os.path.join(data_root, imgName)
        dstPath = os.path.join(save_root,imgName)
        copyfile(imgPath,dstPath)

for r in range(rs):
    if ind_mat[0,r] == 0:
        imgName = nameList[r]
        imgPath = os.path.join(data_root, imgName)
        dstPath = os.path.join(bad_root,imgName)
        copyfile(imgPath,dstPath)


print('Done')