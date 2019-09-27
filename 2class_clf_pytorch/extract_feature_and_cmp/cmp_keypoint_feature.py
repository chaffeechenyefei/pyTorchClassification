#solve the problem of path root of package missing like 'No modules named dataset is found'
import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import cv2
from gunlib.D2Net.featExtract import KeyPointMatch,point2Keypoints,HomoRANSACCompute
from udf.basic import load_obj

pjoin = os.path.join

singletonTest = True

refroot = '/Users/yefeichen/Database/localization/MatterportPanorama/zhgj_perspective/'
queryroot = '/Users/yefeichen/Database/localization/cellphone/zhgj/'
if singletonTest:
    saveroot = '/Users/yefeichen/Database/localization/misspair/'
else:
    saveroot = '/Users/yefeichen/Database/localization/pair/'

imgExt = ['.jpg', '.jpeg', '.bmp', '.png']

queryList = os.listdir(queryroot)
refList = os.listdir(refroot)




imgExt = ['.jpg', '.jpeg', '.bmp', '.png']


for queryname in queryList:
    if singletonTest:
        queryname = 'WechatIMG24.jpeg'
    portQ = os.path.splitext(queryname)
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
                retR = load_obj(feat_d2_name)

                keypointsR = retR['keypoints']
                descR = retR['descriptors']

                kpR = point2Keypoints(keypointsR)

                img3 = KeyPointMatch(kpQ, kpR, descQ, descR, imgQ, imgR, hamming=False)

                ret_flag, M, inlier, matches = HomoRANSACCompute(kpQ, kpR, descQ, descR, hamming=False, flann=False)
                if ret_flag:
                    inlierNum = inlier.sum()
                    print(inlierNum)
                    if inlierNum >= 20 or singletonTest:
                        savename = os.path.join(saveroot, 'd2net_' + portQ[0] + '+' + portR[0] + '++' + str(inlierNum) + '.jpg')
                        cv2.imwrite(savename, img3)
            if singletonTest:
                break
    if singletonTest:
        break


        print('Done')