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
ckpt = 'max_valid_model_.pth'


data_root = '/Users/yefeichen/Database/furniture/ww_furniture_digital/'
# data_root = '/Users/yefeichen/Database/furniture/Material_Matterport/'

print(data_root)
# N_Cls = 109
N_Cls = 26

model_root = Path(model_root)

model = getattr(models, model_name)(
    num_classes=N_Cls,pretrained=False)

model.finetune(os.path.join(model_root,ckpt))
print('weights loaded!')


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

        with torch.no_grad():
            feat,_ = model(inputs)
            feat = feat.squeeze()

        feat = feat.data.cpu().numpy()

        model.get_featuremap()

        # feature_map = model.feature_map.squeeze()

        feature_map = model.feature_map.squeeze().cpu().numpy()

        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() )

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        feature_map = cv2.resize(feature_map, (256,256))
        feature_map = np.expand_dims(feature_map,2)
        feature_map = feature_map.repeat(3,2)

        img = img*feature_map


        # feature_map *= 255
        #
        # feature_map.astype(np.uint8)

        # plt.imshow(feature_map,cmap='gray')
        # plt.close()

        fcnt += 1
        if fcnt % 100 == 0:
            print('Num ' + str(fcnt) + ' processing...')
        npName = _portion[0] + '.jpg'
        npPath = os.path.join(data_root+'tmp',npName)
        cv2.imwrite(npPath,img)


