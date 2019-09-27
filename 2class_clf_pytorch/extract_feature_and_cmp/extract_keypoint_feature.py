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

d2_feat = extractKeyPoints_D2Net(
    model_file='/Users/yefeichen/Desktop/Work/Project/pyTorchClassification/2class_clf_pytorch/result/d2net/d2_tf_no_phototourism.pth')

orb_feat = extractKeyPoints_ORB()

imgroot = '/Users/yefeichen/Database/localization/cellphone/zhgj/'
saveroot= '/Users/yefeichen/Database/localization/cellphone/zhgj/'

imgExt = ['.jpg', '.jpeg', '.bmp', '.png']

imgList = os.listdir(imgroot)

cnt = 0
for imgname1 in imgList:
    port1 = os.path.splitext(imgname1)
    if port1[1] in imgExt:
        img1 = cv2.imread(pjoin(imgroot, imgname1), 1)
        img1 = cv2.resize(img1, (512, 512))
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

        feat1 = d2_feat.detect(img1)
        # feat2 = orb_feat.detect(img1)

        save_d2_name = pjoin(saveroot,'d2_'+port1[0])
        save_obj(feat1,save_d2_name)
        # save_orb_name = pjoin(saveroot, 'orb_' + port1[0])
        # save_obj(feat2,save_orb_name)
        cnt += 1
        if cnt%100 == 1:
            print(str(cnt)+' processed...')

print('Done')
