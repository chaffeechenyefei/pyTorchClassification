#solve the problem of path root of package missing like 'No modules named dataset is found'
import sys
import os
import numpy as np
import pandas as pd
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import cv2
from gunlib.D2Net.featExtract import KeyPointMatch,point2Keypoints,HomoRANSACCompute,AffineRANSACCompute,FundamentalRANSACCompute
from udf.basic import load_obj,save_obj

pjoin = os.path.join

##Path
# refroot = '/Users/yefeichen/Database/localization/MatterportPanorama/zhgj_perspective/'
# queryroot = '/Users/yefeichen/Database/localization/cellphone/zhgj/'
# saveroot = '/Users/yefeichen/Database/localization/non_fund_pair_d2net/'
# csvfileScore = '/Users/yefeichen/Downloads/retrieval_results_score.csv'
# csvfileTopN = '/Users/yefeichen/Downloads/retrieval_results.csv'
refroot = '/home/ubuntu/dataset/indoor/db_indoor/'
queryroot = '/home/ubuntu/dataset/indoor/query_indoor/'
saveroot = '/home/ubuntu/dataset/indoor/tmp/'
csvfileScore = '/home/ubuntu/dataset/indoor/retrieval_results_score.csv'
csvfileTopN = '/home/ubuntu/dataset/indoor/retrieval_results.csv'
##Settings
imgExt = ['.jpg', '.jpeg', '.bmp', '.png']
inlierThreshold = 30
para_t = 70
#CSV
class retrievalClss(object):
    def __init__(self, csvfileScore, csvfileTopN):
        self.pathS = csvfileScore
        self.pathT = csvfileTopN
        self.fdScore = pd.read_csv(csvfileScore)
        self.fdTopn = pd.read_csv(csvfileTopN)
        # get name of cols
        self.ref_img_names = list(self.fdScore.columns)
        pass

    def getTopNNameValue(self, imgQname, topK=1):
        Res = {}
        for i in range(0, int(topK)):
            # i+1 to skip fisrt col which is query image name
            imgRname = self.fdTopn[self.fdTopn['id'] == imgQname].iat[0, i + 1]
            simQR = self.getSimValue(imgQname, imgRname)
            Res[imgRname] = simQR
        return Res

    def getSimValue(self, imgQname, imgRname):
        tmpR = self.fdScore.loc[self.fdScore['id'] == imgQname]
        if len(tmpR) == 0:
            print('No image named', imgQname, 'found')
            return 0
        else:
            return tmpR.iloc[0].at[imgRname]

##Main Program
queryList = os.listdir(queryroot)
retCls = retrievalClss(csvfileScore=csvfileScore,csvfileTopN=csvfileTopN)
Metrics = {}


for queryname in queryList:
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

        dictOfRefImg = retCls.getTopNNameValue(imgQname=queryname,topK=20)
        dictOfRefImgSpv = {}

        for refname in dictOfRefImg.keys():
            portR = os.path.splitext(refname)
            #similarity between query image and reference image
            simQR = dictOfRefImg[refname]

            imgR = cv2.imread(pjoin(refroot, refname), 1)
            imgR = cv2.resize(imgR, (512, 512))

            feat_d2_name = pjoin(refroot, 'd2_' + portR[0])
            if os.path.exists(feat_d2_name + '.pkl'):

                retR = load_obj(feat_d2_name)

                keypointsR = retR['keypoints']
                descR = retR['descriptors']

                kpR = point2Keypoints(keypointsR)

                img3 = KeyPointMatch(kpQ, kpR, descQ, descR, imgQ, imgR, hamming=False)

                ret_flag, M, inlier, matches = FundamentalRANSACCompute(kpQ, kpR, descQ, descR, hamming=False,
                                                                        flann=False)
                if ret_flag:
                    inlierNum = inlier.sum()
                    InLier.append(inlierNum)
                    if inlierNum >= inlierThreshold:
                        savename = os.path.join(saveroot,
                                                'd2_' + portQ[0] + '+' + portR[0] + '++' + str(inlierNum) + '.jpg')
                        cv2.imwrite(savename, img3)

                    spvQR = min(inlierNum,para_t)/para_t
                    dictOfRefImgSpv[refname] = spvQR

        Metrics[queryname] = (dictOfRefImg,dictOfRefImgSpv)

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


save_obj(Metrics,'cmp_Top20_metrics')
print('Done')