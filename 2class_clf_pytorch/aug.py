import imgaug.augmenters as iaa
from scipy.ndimage import affine_transform
from tqdm import tqdm_notebook as tqdm
import numpy as np

from imgaug import augmenters as iaa
import imgaug as ia
import cv2
import os
import numpy as np
import random
import skimage
import math
import pandas as pd
from typing import Union

#=======================================================================================================================
#Adding Backgrounds
#=======================================================================================================================
df_rle = pd.read_csv('/home/ubuntu/dataset/furniture/rles.csv')
df_rle.index = df_rle['Id']
del df_rle['Id']
flick_path = '/home/ubuntu/dataset/furniture/flickr30k_images/flickr30k_images/'
flick_list = os.listdir(flick_path)
flick_list = [l for l in flick_list if 'jpg' in l]
flick_len = len(flick_list)


def do_length_decode(rle, H, W, fill_value=255):
    mask = np.zeros((H,W), np.uint8)
    if type(rle).__name__ == 'float': return mask

    mask = mask.reshape(-1)

    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)

    for r in rle:
        start = r[0]-1
        end = start + r[1]
        mask[start : end] = fill_value
    mask = mask.reshape(W, H).T   # H, W need to swap as transposing.
    return mask

def do_length_encode(x):
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if (b>prev+1): rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    rle = ' '.join([str(r) for r in rle])
    return rle

def get_bbox_rle(rle,img):
    h,w,_ = img.shape
    mask = do_length_decode(rle,h,w,fill_value=1)

    mask = np.where(mask > 0,0,1)
    mask = mask.reshape(h, w, 1)

    hhist = mask.sum(axis=1)
    hhist2 = np.where(hhist>0,1,0)
    hc = hhist2*np.arange(0,h)
    hc = hc.mean() / h
    height = hhist2.sum() / h

    whist = mask.sum(axis=0)
    whist2 = np.where(whist>0,1,0)
    wc = whist2*np.arange(0,w)
    wc = wc.mean() / w
    width = whist2.sum() / w

    return [wc,hc,width,height]

def bbox_scale_pt(bbox:Union[list,tuple],scale:Union[list,tuple]):
    nbbox = list(bbox).copy()
    nbbox[0] = bbox[0]*scale[0]
    nbbox[1] = bbox[1]*scale[1]
    return nbbox

def bbox_offset(bbox:Union[list,tuple],offset:Union[list,tuple]):
    nbbox = list(bbox).copy()
    nbbox[0] = bbox[0] + offset[0]
    nbbox[1] = bbox[1] + offset[1]
    return nbbox

def bbox_scale(bbox:Union[list,tuple],scale:Union[list,tuple]):
    nbbox = list(bbox).copy()
    nbbox[2] = bbox[2]*scale[0]
    nbbox[3] = bbox[3]*scale[1]
    return nbbox

def output_add_bg_img(rle, img, bg):
    bg_h, bg_w, _ = bg.shape
    bg = cv2.resize(bg, (min(bg_w, bg_h), min(bg_w, bg_h)))
    bg_h, bg_w, _ = bg.shape
    h, w, _ = img.shape
    mask = do_length_decode(rle, h, w, fill_value=255)
    mask = np.where(mask > 0, 0, 1)

    img_no_bg = (mask.reshape(h, w, 1) * img).astype('uint8')

    resize_scale = random.uniform(0.8, 1) * bg_h
    if w < h:
        new_h = int(resize_scale)
        new_w = int(resize_scale * w / h)
    else:
        new_w = int(resize_scale)
        new_h = int(resize_scale * h / w)
    img_no_bg_cp = cv2.resize(img_no_bg, (new_w, new_h))
    alpha = np.where(img_no_bg_cp > 0, 1, 0)

    start_x = random.randint(0, bg_w - new_w)
    start_y = random.randint(0, bg_h - new_h)

    bg[start_y:start_y + new_h, start_x:start_x + new_w, :] = \
        bg[start_y:start_y + new_h, start_x:start_x + new_w, :] * (1 - alpha) + \
        img_no_bg_cp
    bg = cv2.medianBlur(bg, 5)
    return bg

def output_add_bg_img_withbbox(rle, bbox ,img, bg):
    bg_h, bg_w, _ = bg.shape
    bg = cv2.resize(bg, (min(bg_w, bg_h), min(bg_w, bg_h)))
    bg_h, bg_w, _ = bg.shape
    h, w, _ = img.shape
    mask = do_length_decode(rle, h, w, fill_value=255)
    mask = np.where(mask > 0, 0, 1)

    img_no_bg = (mask.reshape(h, w, 1) * img).astype('uint8')

    resize_scale = random.uniform(0.3, 1) * bg_h
    if w < h:
        new_h = int(resize_scale)
        new_w = int(resize_scale * w / h)
    else:
        new_w = int(resize_scale)
        new_h = int(resize_scale * h / w)
    img_no_bg_cp = cv2.resize(img_no_bg, (new_w, new_h))
    alpha = np.where(img_no_bg_cp > 0, 1, 0)

    start_x = random.randint(0, bg_w - new_w)
    start_y = random.randint(0, bg_h - new_h)

    # offset = [start_x/bg_w , start_y/bg_h]
    # scale_wh = [new_w/bg_w, new_h/bg_h]

    offset = [start_x / bg_w, start_y / bg_h]
    scale_wh = [new_w / bg_w, new_h / bg_h]

    # (x,y)->(x+offx)/bg_w, (y+offy)/bg_h = x_hat*(new_w/bg_w)+offx/bg_w, y_hat*(new_h/bg_h)+offy/bg_h
    nbbox = bbox_scale_pt(bbox=bbox, scale=scale_wh)
    nbbox = bbox_offset(bbox=nbbox, offset=offset)

    nbbox = bbox_scale(bbox=nbbox, scale=scale_wh)

    bg[start_y:start_y + new_h, start_x:start_x + new_w, :] = \
        bg[start_y:start_y + new_h, start_x:start_x + new_w, :] * (1 - alpha) + \
        img_no_bg_cp
    bg = cv2.medianBlur(bg, 5)
    return bg,nbbox


def rand_bg_resize_crop(image,image_id,imgsize=(256,256)):
    if random.uniform(0, 1) > 0.9:
        if random.uniform(0, 1) > 0.1:#0.1*0.9 = 0.09
            ratio = random.uniform(0.6, 0.99)
            image = cv2.resize(image, imgsize)
            image = random_cropping(image, ratio=ratio, is_random=True)
        else:
            pass #leak of image resize
    else:
        rle = df_rle.loc[image_id].values[0]
        if random.uniform(0, 1) > 0.5:#0.5*0.9 = 0.45
            bg_num = random.randint(0, 48)
            bg = cv2.imread('/home/ubuntu/dataset/furniture/background/' + str(bg_num) + '_bg.png')
        else:#0.45
            bg_num = random.randint(0, flick_len - 1)
            bg = cv2.imread(flick_path + flick_list[bg_num])

        image = output_add_bg_img(rle, image, bg)
        image = cv2.resize(image, imgsize)

        if random.uniform(0, 1) > 0.1:#0.9*0.9 = 0.81
            ratio = random.uniform(0.8, 0.99)
            image = random_cropping(image, ratio=ratio, is_random=True)


    image = cv2.resize(image,imgsize)
    return image

def rand_bg_resize_crop_withbbox(image,image_id,imgsize=(256,256)):
    rle = df_rle.loc[image_id].values[0]
    bbox = get_bbox_rle(rle=rle, img=image)  # [wc,hc,width,height]
    # print(bbox)

    if random.uniform(0, 1) > 0.5:  # 0.5*0.9 = 0.45
        bg_num = random.randint(0, 48)
        bg = cv2.imread('/home/ubuntu/dataset/furniture/background/' + str(bg_num) + '_bg.png')
    else:  # 0.45
        bg_num = random.randint(0, flick_len - 1)
        bg = cv2.imread(flick_path + flick_list[bg_num])

    image,nbbox = output_add_bg_img_withbbox(rle, bbox,image, bg)
    # print(nbbox)
    image = cv2.resize(image, imgsize)

    if random.uniform(0, 1) > 0.1:  # 0.9*0.9 = 0.81
        ratio = random.uniform(0.8, 0.99)
        image,nbbox = random_cropping_withbbox(image, bbox = nbbox,ratio=ratio, is_random=True)

    image = cv2.resize(image,imgsize)
    # print(nbbox)
    return image, nbbox
#=======================================================================================================================
#Adding Backgrounds Ends
#=======================================================================================================================


class RandomErasing(object):

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

def random_erasing(img, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
    if random.uniform(0, 1) > probability:
        return img

    for attempt in range(100):
        area = img.shape[0] * img.shape[1]

        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if w < img.shape[1] and h < img.shape[0]:
            x1 = random.randint(0, img.shape[0] - h)
            y1 = random.randint(0, img.shape[1] - w)
            if img.shape[2] == 3:
                img[x1:x1 + h, y1:y1 + w, 0] = 128
                img[x1:x1 + h, y1:y1 + w, 1] = 128
                img[x1:x1 + h, y1:y1 + w, 2] = 128
            else:
                img[x1:x1 + h, y1:y1 + w, 0] = 128
            return img

    return img


def random_cropping(image, ratio = 0.8, is_random = True):
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]

    zeros = cv2.resize(zeros ,(width,height))
    return zeros

def random_cropping_withbbox(image, bbox = [0.5,0.5,0.5,0.5] ,ratio = 0.8, is_random = True):
    """
    checked with data_aug.ipynb
    :param image: 
    :param bbox: 
    :param ratio: 
    :param is_random: 
    :return: 
    """
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    nbbox = bbox.copy()

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]

    nbbox[0],nbbox[1] = (bbox[0]-start_x/width)/ratio , (bbox[1]-start_y/height)/ratio
    nbbox[2],nbbox[3] = bbox[2]/ratio , bbox[3]/ratio

    zeros = cv2.resize(zeros ,(width,height))
    return zeros,nbbox

def random_flip(image, p=0.5):
    if random.random() < p:
        image = np.flip(image, 1)
    return image

def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

def do_brightness_shift(image, alpha=0.125):
    image = image.astype(np.float32)
    image = image + alpha*255
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def do_contrast(image, alpha=1.0):
    image = image.astype(np.float32)
    gray  = image * np.array([[[0.114, 0.587,  0.299]]]) #rgb to gray (YCbCr)
    gray  = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    image = alpha*image  + gray
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def cropping(image, ratio=0.8, code = 0):
    height, width, _ = image.shape
    target_h = int(height*ratio)
    target_w = int(width*ratio)

    if code==0:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    elif code == 1:
        start_x = 0
        start_y = 0

    elif code == 2:
        start_x = width - target_w
        start_y = 0

    elif code == 3:
        start_x = 0
        start_y = height - target_h

    elif code == 4:
        start_x = width - target_w
        start_y = height - target_h

    elif code == 5:
        return image

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    zeros = cv2.resize(zeros ,(width,height))
    return zeros


def aug_image(image, augment = None):
    flip_code = augment[0]
    crop_code = augment[1]

    if flip_code == 1:
        seq = iaa.Sequential([iaa.Fliplr(1.0)])
        image = seq.augment_image(image)

    image = cropping(image, ratio=0.8, code=crop_code)
    return image


