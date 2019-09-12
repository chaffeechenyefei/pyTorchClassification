#Example of extracting feature using CPU where model is trained by GPU par
import models.models as models
from dataset import *
from utils import (write_event, load_model, mean_df, ThreadingDataLoader as DataLoader,
                   ON_KAGGLE)
from pathlib import Path
import torch
from aug import *
from torch import nn, cuda
import cv2



def torch_load_image(patch):
    HWsize = 256
    image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image ,(HWsize, HWsize))
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = image.reshape([-1, HWsize, HWsize])
    image = image / 255.0
    return torch.FloatTensor(image)


model_root = '/Users/yefeichen/Desktop/Work/Project/pyTorchClassification/2class_clf_pytorch/result/material_toy'
# ckpt = 'model_loss_best6.pt'
# model_name = 'cnntoynet'
model_name = 'cnnvladtoynet'
# model_name = 'resnet18V4'
# model_name = 'resnetvlad18'
ckpt = 'model_'+ model_name +'.pt'


data_root = '/Users/yefeichen/Database/furniture/Material_China FF_E Material_editor_train/'
# data_root = '/Users/yefeichen/Database/furniture/Material_Matterport/'

print(data_root)
# N_Cls = 109
N_Cls = 51

model_root = Path(model_root)

# model = getattr(models, model_name)(
#     num_classes=N_Cls, img_size=256)
model = getattr(models, model_name)(
    num_classes=N_Cls)

# model = torch.nn.DataParallel(model)

use_cuda = cuda.is_available()

if use_cuda:
    model = model.cuda()

model_path = os.path.join(model_root,ckpt)
load_model(model,model_path)


print('weights loaded!')
# targets = torch.rand(N_Cls,1)*10
# targets = targets.int()%10

fcnt = 0
ecnt = 0

fileList = os.listdir(data_root)

for _file in fileList:
    _portion = os.path.splitext(_file)
    if _portion[1] in ['.jpg','.png','.jpeg']:
        imgName = _file
        imgPath = os.path.join(data_root,imgName)
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
            feat,_ = model(inputs)
            feat = feat.squeeze()

        feat = feat.data.cpu().numpy()

        fcnt += 1
        if fcnt % 100 == 0:
            print('Num ' + str(fcnt) + ' processing...')
        npName = _portion[0] + '.npy'
        npPath = os.path.join(data_root,npName)
        np.save(npPath, feat)


