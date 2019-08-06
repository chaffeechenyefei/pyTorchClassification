import numpy as np
import os
import yolo_utils
import argparse
from itertools import islice
import json
from pathlib import Path
import shutil
import warnings
from typing import Dict
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.exceptions import UndefinedMetricWarning
import torch
from torch import nn, cuda
from torch.optim import Adam, SGD
import tqdm
import models.models as models
from dataset import *
from transforms import train_transform, test_transform
from utils import (write_event, load_model, mean_df, ThreadingDataLoader as DataLoader,
                   ON_KAGGLE)
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

#add clf model
#func
def torch_load_image(patch):
    image = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
    image = aug_image(image, augment = tta_code)
    image = cv2.resize(image ,(256, 256))

    image = np.transpose(image, (2, 0, 1))
    image = image.astype(np.float32)
    image = image.reshape([-1, 256, 256])
    image = image / 255.0
    return torch.FloatTensor(image)
#para
run_root = 'result/se50_talking_0.8'
ckpt = 'model_loss_best.pt'
model_name = 'se_resnext50'

run_root = Path(run_root)
model = getattr(models, model_name)(
            num_classes=N_CLASSES, pretrained='imagenet')
model = torch.nn.DataParallel(model)
model = model.cuda()
load_model(model, Path(str(run_root) + '/' + ckpt))
print('weights loaded!')
test_root = DATA_ROOT + '/test2'

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

#add clf model


det_model_root = '/Users/yefeichen/Models/peopleDetection/TA7/'
# det_config = det_model_root + 'yolov3-tiny_3l_person.cfg'
# det_weights = det_model_root + '/weights/yolov3-tiny_3l_person_final.weights'
det_config = det_model_root + 'yolov3_person.cfg'
det_weights = det_model_root + 'yolov3_person_16000.weights'

cls_model_root = '/Users/yefeichen/Models/classification/cls7/'
cls_config = cls_model_root + 'wwdarknet53v3.cfg'
# cls_weights = cls_model_root + 'wwdarknet53_train2_best.dat'
cls_weights = cls_model_root + 'wwdarknet53v3_60000.weights'


labelFile = cls_model_root + 'activity_wework.names'

# fileDir = '/Users/yefeichen/Database/wework_activity/Scene_4/test_2/'
fileDir = '/Users/yefeichen/Database/wwa_new/activity_7_classes/medium_people/test_sample/'
outfileDir = '/Users/yefeichen/Database/wwa_new/activity_7_classes/medium_people/test_result/'

# outfileDir = '/Users/yefeichen/Database/wework_activity/Scene_4/result_2stage_full_TA7/'

prob_thresh = 0.15
cls_thresh = 0.14
nms_threh = 0.45


def sigmoid(mat):
    exp_mat = np.exp(mat * -1)
    sigm_mat = 1 + exp_mat

    one_mat = np.ones(mat.shape)

    res = one_mat / sigm_mat

    return res



det_net = cv2.dnn.readNetFromDarknet(det_config,det_weights)
det_layer_names = det_net.getLayerNames()
det_layer_names = [det_layer_names[i[0] - 1] for i in det_net.getUnconnectedOutLayers()]


colors = [[0,255,0]]


labels = open(labelFile).read().strip().split('\n')

fcnt = 0
files = os.listdir(fileDir)
for filename in files:
    portion = os.path.splitext(filename)
    if portion[1] == '.jpg':
        fcnt += 1
        if (fcnt%100 == 0):
            print(str(fcnt) + ' done\n' )
        img = cv2.imread(fileDir+filename, cv2.IMREAD_COLOR)
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (704, 704),
                                    swapRB=True, crop=False)

        # Perform a forward pass of the YOLO object detector
        det_net.setInput(blob)

        # Getting the outputs from the output layers
        det_outs = det_net.forward(det_layer_names)
        det_boxes, det_confidences, det_classids = yolo_utils.generate_boxes_confidences_classids(det_outs, height, width, 0.15)

        # Apply Non-Maxima Suppression to suppress overlapping bounding boxes
        det_idxs = cv2.dnn.NMSBoxes(det_boxes, det_confidences, prob_thresh, nms_threh)

        patches = yolo_utils.get_boxes(img,det_boxes,det_idxs,1.4)

        preds = []
        for patch in patches:
            patch = torch_load_image(patch)
            patch = patch.cuda()
            pred = torch.sigmoid(model(inputs).squeeze())
            pred = pred.data.cpu().numpy()
            pred[pred<cls_thresh] = 0
            preds.append(pred)


        yolo_utils.draw_labels_and_boxes_2_stage(img,det_boxes,det_idxs,preds,labels)

        cv2.imwrite(outfileDir+filename,img)

        # yolo_utils.show_image(img)








