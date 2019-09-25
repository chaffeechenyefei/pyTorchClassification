import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt

from gunlib.D2Net.d2Net_predict import D2Net
from gunlib.D2Net.utils import preprocess_image
from gunlib.D2Net.pyramid import process_multiscale


# CUDA
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
pjoin = os.path.join


class extractKeyPoints_D2Net(object):
    def __init__( self,
                  model_file='d2_tf_no_phototourism.pth',
                  use_relu=True,
                  img_size = 512):
        self.model = D2Net(
            model_file=model_file,
            use_relu=use_relu,
            use_cuda=use_cuda
        )
        self.img_size = img_size

    def detect(self,image, multiscale=False, nratio=0.3):
        resized_image = cv2.resize(image,(self.img_size,self.img_size)).astype(np.float32)
        fact_i = image.shape[0] / resized_image.shape[0]
        fact_j = image.shape[1] / resized_image.shape[1]

        input_image = preprocess_image(
            resized_image,
            preprocessing='torch'
        )

        with torch.no_grad():
            if multiscale:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    self.model
                )
            else:
                keypoints, scores, descriptors = process_multiscale(
                    torch.tensor(
                        input_image[np.newaxis, :, :, :].astype(np.float32),
                        device=device
                    ),
                    self.model,
                    scales=[1]
                )

        # Input image coordinates
        keypoints[:, 0] *= fact_i #height
        keypoints[:, 1] *= fact_j #width
        # i, j -> u, v
        keypoints = keypoints[:, [1, 0, 2]] #x,y,scale

        rk = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

        nfeatures = int(len(scores)*nratio)

        rk = rk[:nfeatures-1]

        keypoints = keypoints[rk,:]
        descriptors = descriptors[rk,:]
        scores = scores[rk]

        return {
            'keypoints': keypoints, #[N,3]
            'scores': scores, #[N,]
            'descriptors': descriptors #[N,feat_dim]
        }


class extractKeyPoints_ORB(object):
    def __init__(self,nfeatures=1000,img_size = 512):
        self.model = cv2.ORB_create(nfeatures=nfeatures,scoreType=cv2.ORB_FAST_SCORE)
        self.img_size = img_size

    def detect(self,image):
        resized_image = cv2.resize(image,(self.img_size,self.img_size))
        fact_i = image.shape[0] / resized_image.shape[0] #height
        fact_j = image.shape[1] / resized_image.shape[1] #width

        keypoints, descriptors = self.model.detectAndCompute(resized_image,None)
        keypoints = cv2.KeyPoint_convert(keypoints)

        keypoints[:, 0] *= fact_j
        keypoints[:, 1] *= fact_i

        return {
            'keypoints':keypoints, #[N,2]
            'descriptors':descriptors #[N,feat_dim]
        }

def point2Keypoints(pts):
    """
    convert np.array[N,2] -> list[cv2.KeyPoint]
    :param pts: 
    :return: 
    """
    kypt = []
    for pt in pts:
        ptt = cv2.KeyPoint(x=pt[0], y=pt[1], _size=1)
        kypt.append(ptt)
    return kypt

def HomoRANSACCompute(kp1,kp2,desc1,desc2,hamming=False,flann=False):
    MIN_MATCH_COUNT = 10
    if not flann:
        if not hamming:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        else:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(desc1, desc2)
        # matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches
                                  ]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches
                                  ]).reshape(-1, 1, 2)

            M, inlierMask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return True, M, inlierMask, matches
        else:
            print('Two few matched points')
            return False,None,None,None
    else:
        FLANN_INDEX_KDITREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)

        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good
                                  ]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good
                                  ]).reshape(-1, 1, 2)

            M, inlierMask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            return True,M,inlierMask,good
        else:
            print('Two few matched points')
            return False,None,None,None

def KeyPointMatch(kp1,kp2,desc1,desc2,img1,img2,num_features = 100):
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       flags=0)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:num_features-1], None, **draw_params)
    return img3



if __name__ == '__main__':
    d2_feat = extractKeyPoints_D2Net(model_file='/Users/yefeichen/Desktop/Work/Project/pyTorchClassification/2class_clf_pytorch/result/d2net/d2_tf_no_phototourism.pth')
    # imgroot = '/Users/yefeichen/Desktop/Work/Project/d2-net/qualitative/images/pair_1/'
    # imgroot = '/Users/yefeichen/Desktop/Work/Project/PanoramaImageViewer/'
    imgroot = '/Users/yefeichen/Downloads/'
    img1 = cv2.imread( pjoin(imgroot,'China-Overseas-International-Center-09192019_095232.jpg'), 1 )
    img2 = cv2.imread( pjoin(imgroot,'China-Overseas-International-Center-09192019_095256.jpg'), 1 )

    img1 = cv2.resize(img1,(512,512))
    img2 = cv2.resize(img2,(512, 512))


    ret1 = d2_feat.detect(img1)
    ret2 = d2_feat.detect(img2)

    keypoints1 = ret1['keypoints']
    desc1 = ret1['descriptors']
    keypoints2 = ret2['keypoints']
    desc2 = ret2['descriptors']

    kp1 = point2Keypoints(keypoints1)
    kp2 = point2Keypoints(keypoints2)

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       flags=0)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = np.array([])
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, **draw_params)
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

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = np.array([])
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:100], None, **draw_params)
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