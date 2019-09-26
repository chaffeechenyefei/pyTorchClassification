import numpy as np
import torch
import os
import cv2
from gunlib.D2Net.featExtract import *

# d2_feat = extractKeyPoints_D2Net(model_file='/Users/yefeichen/Desktop/Work/Project/pyTorchClassification/2class_clf_pytorch/result/d2net/d2_tf_no_phototourism.pth')
d2_feat = extractKeyPoints_D2Net(
    model_file='/home/ubuntu/git/pyTorchClassification/2class_clf_pytorch/result/d2_tf_no_phototourism.pth')
# imgroot = '/Users/yefeichen/Desktop/Work/Project/d2-net/qualitative/images/pair_1/'
# imgroot = '/Users/yefeichen/Desktop/Work/Project/PanoramaImageViewer/'
imgroot = '/home/ubuntu/git/pyTorchClassification/2class_clf_pytorch/result/'
img1 = cv2.imread(pjoin(imgroot, 'China-Overseas-International-Center-09192019_095232.jpg'), 1)
img2 = cv2.imread(pjoin(imgroot, 'China-Overseas-International-Center-09192019_095256.jpg'), 1)

img1 = cv2.resize(img1, (512, 512))
img2 = cv2.resize(img2, (512, 512))

ret1 = d2_feat.detect(img1)
ret2 = d2_feat.detect(img2)

keypoints1 = ret1['keypoints']
desc1 = ret1['descriptors']
keypoints2 = ret2['keypoints']
desc2 = ret2['descriptors']

kp1 = point2Keypoints(keypoints1)
kp2 = point2Keypoints(keypoints2)

img3 = KeyPointMatch(kp1, kp2, desc1, desc2, img1, img2, hamming=False)
cv2.imwrite('d2net_debug.jpeg', img3)

ret_flag, M, inlier, matches = HomoRANSACCompute(kp1, kp2, desc1, desc2, hamming=False, flann=False)

print(inlier.sum())

orb_feat = extractKeyPoints_ORB()
ret1 = orb_feat.detect(img1)
ret2 = orb_feat.detect(img2)

keypoints1 = ret1['keypoints']
desc1 = ret1['descriptors']
keypoints2 = ret2['keypoints']
desc2 = ret2['descriptors']

kp1 = point2Keypoints(keypoints1)
kp2 = point2Keypoints(keypoints2)

img3 = KeyPointMatch(kp1, kp2, desc1, desc2, img1, img2, hamming=True)
cv2.imwrite('orb_debug.jpeg', img3)

ret_flag, M, inlier, matches = HomoRANSACCompute(kp1, kp2, desc1, desc2, hamming=True, flann=False)
print(inlier.sum())

print('Done')

# for kypt in keypoints:
#     pt = (int(kypt[0]),int(kypt[1]))
#     cv2.circle(raw_img,pt,3,(255,0,0),2)
#
# print(ret['descriptors'].shape)
#
# cv2.imwrite('debug.jpeg',raw_img)