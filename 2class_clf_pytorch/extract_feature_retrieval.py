#Example of extracting feature using CPU where model is trained by GPU par
import models.models as models
from dataset import *
from utils import (write_event, load_par_gpu_model_cpu, mean_df, ThreadingDataLoader as DataLoader,
                   ON_KAGGLE)
from pathlib import Path
import torch
from aug import *
from torch import nn, cuda
import cv2
from sklearn.metrics.pairwise import euclidean_distances



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

data_root = '/Users/yefeichen/Database/furniture/chair_from_digital/'
test_root = '/Users/yefeichen/Database/furniture/collect_from_matterport_chair/'
N_Cls = 109

model_root = Path(model_root)

model = getattr(models, model_name)(
            num_classes=N_Cls, pretrained='imagenet')

# model = torch.nn.DataParallel(model)

use_cuda = cuda.is_available()

if use_cuda:
    model = model.cuda()

model_path = os.path.join(model_root,ckpt)
load_par_gpu_model_cpu(model,model_path)


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

# targets = torch.rand(N_Cls,1)*10
# targets = targets.int()%10
#testing
testList = os.listdir(test_root)
for _file in testList:
    if '.png' in _file: #'_s.jpg'
        imgName = _file
        imgPath = os.path.join(test_root,imgName)
        img = cv2.imread(imgPath)
        try:
            img.shape
        except:
            print('Err:' + imgName)
            ecnt += 1
            continue

        inputs = torch_load_image(img)
        if use_cuda:
            inputs = inputs.cuda()

        inputs = torch.unsqueeze(inputs,0)

        with torch.no_grad():
            feat,_= model(inputs)
            feat = feat.squeeze()

        feat = feat.data.cpu().numpy()

        # pairDist = cosDist(trFeat,feat.reshape(1,-1))
        pairDist = euclidean_distances(feat.reshape(1,-1),trFeat)
        pairDist = pairDist.squeeze()
        # idx = np.argmax(pairDist).item()
        #idxs = pairDist.argsort()[-3:][::-1]
        #argsort() 从小到大排列
        idxs = pairDist.argsort()[:3]
        # idx = np.argmin(pairDist, axis=1).item()

        img3 = np.zeros((256, 256 * 4, 3), dtype=np.float32)
        img1 = cv2.resize(img, (256, 256))
        img3[:, 0:256, :] = img1

        kt = 1

        for idx in idxs:
            simName = os.path.join(data_root,nameList[idx])
            imgFromTr = cv2.imread(simName)
            img2 = cv2.resize(imgFromTr,(256,256))
            img3[:,256*kt:256*(kt+1),:] = img2
            kt = kt+1

        _portion = os.path.splitext(_file)
        savePath = os.path.join(test_root,'pair/'+_portion[0]+'_cmp.jpg')
        print(savePath)
        cv2.imwrite(savePath,img3)
