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



def torch_load_image(patch):
    image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    image = cropping(image,ratio=0.9)
    image = cv2.resize(image ,(256, 256))
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = image.reshape([-1, 256, 256])
    image = image / 255.0
    return torch.FloatTensor(image)


model_root = '/Users/yefeichen/Desktop/Work/Project/pyTorchClassification/2class_clf_pytorch/result/furniture_toy'
ckpt = 'model_loss_best.pt'
model_name = 'resnet50V4'

data_root = '/Users/yefeichen/Database/furniture/collect_from_matterport_chair/'
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


