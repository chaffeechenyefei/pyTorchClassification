#solve the problem of path root of package missing like 'No modules named dataset is found'
import sys
import os
import numpy as np
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import cv2
from gunlib.D2Net.featExtract import KeyPointMatch,point2Keypoints,HomoRANSACCompute,AffineRANSACCompute,FundamentalRANSACCompute
from udf.basic import load_obj

pjoin = os.path.join

singletonTest = False

refroot = '/Users/yefeichen/Database/localization/MatterportPanorama/shanghai_tower_perspective/'
queryroot = '/Users/yefeichen/Database/localization/cellphone/zhgj/'
if singletonTest:
    saveroot = '/Users/yefeichen/Database/localization/misspair/'
else:
    saveroot = '/Users/yefeichen/Database/localization/non_fund_pair_d2net/'

imgExt = ['.jpg', '.jpeg', '.bmp', '.png']

queryList = os.listdir(queryroot)
refList = os.listdir(refroot)

inlierThreshold = 30


imgExt = ['.jpg', '.jpeg', '.bmp', '.png']


for queryname in queryList:
    if singletonTest:
        queryname = 'WechatIMG24.jpeg'
    portQ = os.path.splitext(queryname)

    InLier = []

    if portQ[1] in imgExt:
        imgQ = cv2.imread(pjoin(queryroot, queryname), 1)
        imgQ = cv2.resize(imgQ, (512, 512))

        feat_d2_name = pjoin(queryroot, 'd2_' + portQ[0])
        retQ = load_obj(feat_d2_name)

        keypointsQ = retQ['keypoints']
        descQ = retQ['descriptors']
        kpQ = point2Keypoints(keypointsQ)

        for refname in refList:
            if singletonTest:
                refname = 'China-Overseas-International-Center-09262019_171912_theta@-120_fov@90_phi@0.jpg'
            portR = os.path.splitext(refname)
            if portR[1] in imgExt:
                imgR = cv2.imread(pjoin(refroot, refname), 1)
                imgR = cv2.resize(imgR, (512, 512))

                feat_d2_name = pjoin(refroot, 'd2_' + portR[0])
                if os.path.exists( feat_d2_name+'.pkl'):

                    retR = load_obj(feat_d2_name)

                    keypointsR = retR['keypoints']
                    descR = retR['descriptors']

                    kpR = point2Keypoints(keypointsR)

                    img3 = KeyPointMatch(kpQ, kpR, descQ, descR, imgQ, imgR, hamming=False)

                    ret_flag, M, inlier, matches = FundamentalRANSACCompute(kpQ, kpR, descQ, descR, hamming=False, flann=False)
                    if ret_flag:
                        inlierNum = inlier.sum()
                        # print(inlierNum)
                        InLier.append(inlierNum)
                        if inlierNum >= inlierThreshold or singletonTest:
                            savename = os.path.join(saveroot, 'd2_' + portQ[0] + '+' + portR[0] + '++' + str(inlierNum) + '.jpg')
                            cv2.imwrite(savename, img3)
            if singletonTest:
                break
        InLier.sort(reverse=True)

        if InLier is not None and InLier != []:
            maxInLierNum = InLier[0]
            InLierOver20 = [v for v in InLier if v >= inlierThreshold]
            numInLierImg = len(InLierOver20)
            avgInLierNum = np.array(InLierOver20)
            avgInLierNum = avgInLierNum.mean()
        else:
            maxInLierNum = 0
            numInLierImg = 0
            avgInLierNum = 0

        print( queryname + ':' + 'maxInLierNum='+ str(maxInLierNum)
               + ' numInLierImg=' + str(numInLierImg)
               + ' avgInLierNum=' + str(avgInLierNum) )

        if singletonTest:
            break


print('Done')