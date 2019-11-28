from skimage.util import view_as_windows as viewW
import numpy as np
import pickle
import time

#=======================================================================================================================
# list -> str
#=======================================================================================================================
def list2str(List:list,backend = '')->str:
    """
    list2str
    :param List: input list
    :param backend: define the end of line '\n','.'
    :return: string of list with ',' as split
    """
    res = ''
    for ele in List:
        res += str(ele) + ','
    if len(res) > 0:
        res = res[:-1]
    return res+backend

#=======================================================================================================================
# different version of image to column(img2col,im2col)
#=======================================================================================================================
#https://stackoverflow.com/questions/30109068/implement-matlabs-im2col-sliding-in-python
# im2col
def im2col_sliding_broadcasting(A, BSZ, stepsize=1):
    """
    im2col_sliding_broadcasting
    :param A: [H,W]
    :param BSZ: [winH,winW]
    :param stepsize: 
    :return: [winH*winW,N]
    """
    # Parameters
    M,N = A.shape
    col_extent = N - BSZ[1] + 1
    row_extent = M - BSZ[0] + 1

    # Get Starting block indices
    start_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:,None]*N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    return np.take (A,start_idx.ravel()[:,None] + offset_idx.ravel()[::stepsize])


def im2col_sliding_strided(A, BSZ, stepsize=1):
    # Parameters
    m,n = A.shape
    s0, s1 = A.strides
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1

    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]


def im2col_sliding_strided_v2(A, BSZ, stepsize=1):
    return viewW(A, (BSZ[0],BSZ[1])).reshape(-1,BSZ[0]*BSZ[1]).T[:,::stepsize]

#=======================================================================================================================
# save almost any objects into file
#=======================================================================================================================
def save_obj(obj, name ):
    with open( name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#=======================================================================================================================
# topk accuracy of numpy version
#=======================================================================================================================
def calc_topk_acc(QRscore, y_truth, k=3):
    """
    QRscore: similarity score matrix shape [Q,R]
    y_truth: index(related with R) of truth label of Query
    """
    max_k_preds = QRscore.argsort(axis=1)[:, -k:][:, ::-1]  # 得到top-k max label
    match_array = np.logical_or.reduce(max_k_preds == y_truth, axis=1)  # 得到匹配结果
    topk_acc_score = match_array.sum() / match_array.shape[0]
    return topk_acc_score

def calc_topk_acc_cat_all(QRscore,y_truth_cat,R_cat,k=3):
    """
    QRscore: similarity score matrix shape [Q,R]
    y_truth: index(related with R) of truth label of Query
    return: list oftop1-topk acc
    """
    res = []
    y_truth_cat = y_truth_cat.reshape(-1,1)
    max_k_preds = QRscore.argsort(axis=1)[:, -k:][:, ::-1] #得到top-k max label
    max_k_cat = R_cat[max_k_preds]
    M = max_k_cat==y_truth_cat
    for k in range(M.shape[1]):
        match_array = np.logical_or.reduce(M[:,:k+1], axis=1) #得到匹配结果
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)
    return res

#=======================================================================================================================
# topk recall score of numpy version
#=======================================================================================================================
def topk_recall_score(pred,truth,topk=5):
    """
    N: sample number
    M: class number
    pred [N,M] 
    truth [N,M]
    large value of score means 1 in truth, while small value of score means 0 in truth.
    """
    N,M = pred.shape
    _N,_M = truth.shape
    assert( M==_M and N==_N )
    #get rank of the predicted score, acsending
    rank_of_pred = np.argsort(pred,axis=1)
    #get topk ranked label
    ranked_label = np.take_along_axis(truth,rank_of_pred,axis=1)[:,-1:-topk-1:-1]
    recall_score = ranked_label.sum()/topk/N
    return recall_score

def topk_recall_score_all(pred,truth,topk=5):
    """
    N: sample number
    M: class number
    pred [N,M] 
    truth [N,M]
    large value of score means 1 in truth, while small value of score means 0 in truth.
    return each topL(L<=topk) average recall_score 
    """
    N,M = pred.shape
    _N,_M = truth.shape
    assert( M==_M and N==_N )
    #get rank of the predicted score, acsending
    rank_of_pred = np.argsort(pred,axis=1)
    #get topk ranked label
    ranked_label = np.take_along_axis(truth,rank_of_pred,axis=1)[:,-1:-topk-1:-1]#[N,topk]
    nTruth = truth.sum(axis=1).reshape(-1,1)#[N,1]
    nTruth[nTruth<1]=1
    ranked_label = ranked_label/nTruth
    recall_score = np.cumsum(ranked_label.sum(axis=0).reshape(1,-1),axis=1)
    # nTruth = truth.sum(axis=1).reshape(1,-1)
    # nTop = np.array(range(topk)).reshape(1,topk) + 1
    recall_score = (recall_score / N)
    return recall_score

class timer(object):
    def __init__(self,it:str='',display=True):
        self._start = 0
        self._end = 0
        self._name = it
        self._display = display
        pass

    def start(self,it:str):
        self._start = time.time()
        if it is not None:
            self._name = it
        else:
            pass

    def end(self):
        self._end = time.time()

    def diff(self):
        tt = self._end-self._start
        return tt

    def eclapse(self):
        self.end()
        tt = self.diff()
        if self._display:
            print('<<%s>> eclapse: %f sec...'%(self._name,tt))


#=======================================================================================================================
# main
#=======================================================================================================================
if __name__ == '__main__':
    import os
    import cv2

    src_path = '/Users/yefeichen/Database/furniture/Material_China FF_E Material_editor_train/'
    dst_path = '/Users/yefeichen/Database/furniture/Material_China FF_E Material_editor_train_small/'

    pjoin = os.path.join

    Bw = [256, 256]
    Bn = 20

    filelist = os.listdir(src_path)

    cls = -1

    for file in filelist:
        portion = os.path.splitext(file)
        if portion[1] in ['.png', '.jpg']:
            cnt = 0
            cls += 1
            img = cv2.imread(pjoin(src_path, file))
            h, w, c = img.shape
            stride = 150
            ch_patches = []
            for i in range(c):
                patches = im2col_sliding_broadcasting(img[:, :, i], Bw, stride).reshape(Bw[0], Bw[1], -1)
                ch_patches.append(patches)
            _, _, L = ch_patches[0].shape
            for j in range(L):
                nimg = np.zeros((Bw[0], Bw[1], c), dtype=np.uint8)
                for i in range(c):
                    nimg[:, :, i] = ch_patches[i][:, :, j]
                    #             plt.imshow(nimg)
                save_name = portion[0] + '_cls_' + str(cls) + '_' + str(cnt) + '.jpg'
                cv2.imwrite(pjoin(dst_path, save_name), nimg)
                cnt += 1
            print(file + 'done...')