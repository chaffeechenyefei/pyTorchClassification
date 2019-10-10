#solve the problem of path root of package missing like 'No modules named dataset is found'
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import cv2
from gunlib.D2Net.featExtract import extractKeyPoints_D2Net,extractKeyPoints_ORB
from udf.basic import save_obj

pjoin = os.path.join

on_aws = True

if on_aws:
    model_file = '/home/ubuntu/CV/models/d2net/d2_tf_no_phototourism.pth'
else:
    model_file = '/Users/yefeichen/Desktop/Work/Project/pyTorchClassification/2class_clf_pytorch/result/d2net/d2_tf_no_phototourism.pth'

d2_feat = extractKeyPoints_D2Net(
    model_file= model_file)

orb_feat = extractKeyPoints_ORB()

if on_aws:
    imgroot =  '/home/ubuntu/dataset/indoor/db_indoor/'
else:
    imgroot = '/Users/yefeichen/Database/localization/MatterportPanorama/shanghai_tower_perspective/'
saveroot= imgroot

imgExt = ['.jpg', '.jpeg', '.bmp', '.png']

imgList = os.listdir(imgroot)

cnt = 0
for imgname1 in imgList:
    port1 = os.path.splitext(imgname1)
    # imgname1 = 'China-Overseas-International-Center-09262019_172832_theta@-90_fov@90_phi@0.jpg'
    if port1[1] in imgExt:
        img1 = cv2.imread(pjoin(imgroot, imgname1), 1)
        img1 = cv2.resize(img1, (512, 512))
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

        feat1 = d2_feat.detect(img1)
        # try:
        #     feat2 = orb_feat.detect(img1)
        # except:
        #     print(imgname1)
        #     continue

        save_d2_name = pjoin(saveroot,'d2_'+port1[0])
        save_obj(feat1,save_d2_name)
        # if feat2['flag']:
        #     save_orb_name = pjoin(saveroot, 'orb_' + port1[0])
        #     save_obj(feat2,save_orb_name)

        cnt += 1
        if cnt%100 == 1:
            print(str(cnt)+' processed...')

print('Done')
