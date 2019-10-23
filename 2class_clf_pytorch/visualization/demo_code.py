#Example of extracting feature using CPU where model is trained by GPU par
import models.models as models
from dataset import *
import cv2
from utils import *

import sys
import os
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def torch_load_image(patch):
    HWsize = 256
    image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image ,(HWsize, HWsize))
    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = image.reshape([-1, HWsize, HWsize])
    image = image / 255.0
    return torch.FloatTensor(image)


model_root = '/Users/yefeichen/Desktop/Work/Project/pyTorchClassification/2class_clf_pytorch/result/furniture_inception_yiheng/'
model_name = 'inception_v4'
ckpt = 'max_valid_model_7146.pth'


data_root = '/Users/yefeichen/Database/furniture/ww_furniture_digital/'

print(data_root)
N_Cls = 26

model_root = Path(model_root)

model = getattr(models, model_name)(
    num_classes=N_Cls,pretrained=False)

model.finetune(os.path.join(model_root,ckpt))
print('weights loaded!')

if use_cuda:
    model.cuda()


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
        inputs = torch.unsqueeze(inputs,0)

        if use_cuda:
            inputs = inputs.cuda()

        with torch.no_grad():
            feat,_ = model(inputs)
            feat = feat.squeeze()

        #feature of image
        feat = feat.data.cpu().numpy()


