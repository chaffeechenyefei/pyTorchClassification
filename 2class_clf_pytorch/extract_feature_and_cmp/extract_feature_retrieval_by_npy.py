#Example of extracting feature using CPU where model is trained by GPU par
from sklearn.metrics.pairwise import euclidean_distances,cosine_distances

#solve the problem of path root of package missing like 'No modules named dataset is found'
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import models.models as models
from dataset import *


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


dataSet = [
    ('/Users/yefeichen/Database/furniture/collect_from_matterport_all0823/','.png'),#0
    ('/Users/yefeichen/Database/furniture/chair_from_digital/','.jpg'),             #1
    ('/Users/yefeichen/Database/furniture/collect_from_matterport_chair/','.png'),  #2
    ('/Users/yefeichen/Database/furniture/kkkk/','.jpg'),                           #3
    ('/Users/yefeichen/Database/furniture/Material_China FF_E Material_editor_train/','.jpg'), #4
    ('/Users/yefeichen/Database/furniture/Material_Matterport/','.png'), #5
    ('/Users/yefeichen/Desktop/Work/Project/PanoramaImageViewer/ww_sample/','.jpg'), #6
           ]

topK = 5

data_root,data_root_ext = dataSet[6]
test_root,test_root_ext = dataSet[6]
# N_Cls = 109


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
        nameList.append(nameSp[0]+data_root_ext)

trFeat = np.vstack(featList)

print('Features loaded from data: ' + str(trFeat.shape) )

# targets = torch.rand(N_Cls,1)*10
# targets = targets.int()%10
#testing
testList = os.listdir(test_root)
all_img = []
for _file in testList:
    if test_root_ext in _file: #'_s.jpg'
        imgName = _file
        imgPath = os.path.join(test_root,imgName)
        img = cv2.imread(imgPath)
        try:
            img.shape
        except:
            print('Err:' + imgName)
            ecnt += 1
            continue
        portion = os.path.splitext(_file)
        npPath = os.path.join(test_root, portion[0]+'.npy')
        feat = np.load(npPath).reshape(1,-1)

        # pairDist = cosDist(trFeat,feat.reshape(1,-1))
        pairDist = euclidean_distances(feat.reshape(1,-1),trFeat)
        # pairDist = cosine_distances(feat.reshape(1,-1),trFeat)
        pairDist = pairDist.squeeze()
        # idx = np.argmax(pairDist).item()
        #idxs = pairDist.argsort()[-3:][::-1]
        #argsort() 从小到大排列
        idxs = pairDist.argsort()[:topK]
        # idx = np.argmin(pairDist, axis=1).item()

        img3 = np.zeros((256, 256 * (topK+1), 3), dtype=np.float32)
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
        savePath = os.path.join(test_root,'pair/'+_portion[0]+ '_' + str(pairDist[idxs[1]]) + '_.jpg')
        print(savePath)
        cv2.imwrite(savePath,img3)

        all_img.append(img3)


# all_img = np.concatenate(all_img,axis=0)
# savePath = os.path.join(test_root,'pair/all_cmp.jpg')
# cv2.imwrite(savePath,all_img)
