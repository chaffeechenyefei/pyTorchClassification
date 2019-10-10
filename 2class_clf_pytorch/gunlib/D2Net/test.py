import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
from gunlib.D2Net.featExtract import *
import pandas as pd

d2_feat = extractKeyPoints_D2Net(model_file='/Users/yefeichen/Desktop/Work/Project/pyTorchClassification/2class_clf_pytorch/result/d2net/d2_tf_no_phototourism.pth')

f = pd.read_csv('/Users/yefeichen/Downloads/rgbd_yanlin/1.csv',header=None)
W = np.array(f[0]).max()+1
H = np.array(f[1]).max()+1
print('image',H,W)
D = np.array([f[2],f[3],f[4]])
D = D.transpose()
D = D.reshape(W,H,3)
D = np.transpose(D,(1,0,2))

imgR = cv2.imread('/Users/yefeichen/Downloads/rgbd_yanlin/1.jpg_Perspective.png')
imgQ = cv2.imread('/Users/yefeichen/Downloads/rgbd_yanlin/view1.jpg')

imgQ = cv2.resize(imgQ,(512,512))


retQ = d2_feat.detect(imgQ)
retR = d2_feat.detect(imgR)

keypointsQ = retQ['keypoints']
descQ = retQ['descriptors']
keypointsR = retR['keypoints']
descR = retR['descriptors']


kpQ = point2Keypoints(keypointsQ)
kpR = point2Keypoints(keypointsR)

img3 = KeyPointMatch(kpQ,kpR,descQ,descR,imgQ,imgR,hamming=False,num_features=20)
cv2.imwrite('_debug.jpg',img3)

ret_flag, M, inlier, matches = HomoRANSACCompute(kpQ, kpR, descQ, descR, hamming=False, flann=False)

print(len(matches),inlier.shape)
# (matches[0].trainIdx, matches[0].queryIdx)
qKp = []
for i,m in enumerate(matches):
    if inlier[i]==1:
        qKp.append(keypointsQ[m.queryIdx])

V = np.stack(qKp)
V = V[:,0:2]
#
V = V.astype(np.int32)

W = []
for i,pt in enumerate(V):
#     print(pt)
    W.append(D[pt[1],pt[0],:])
W = np.stack(W)
W = W[:,:,np.newaxis]
print(W.shape)

V = V[:,:,np.newaxis]

W = W.astype(np.float32)
V = V.astype(np.float32)

focal_length = 512
center = (512/2,512/2)
cameraMatrix = np.array(
        [[focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]], dtype = np.float32)

distCoeffs = np.zeros((5,1)).astype(np.float32)
print(W.dtype,V.dtype,cameraMatrix.dtype,distCoeffs.dtype)
ret,R,T = cv2.solvePnP(W,V,cameraMatrix,distCoeffs,flags = cv2.SOLVEPNP_ITERATIVE)

print(T)
print('Done')