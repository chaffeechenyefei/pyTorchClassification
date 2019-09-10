from skimage.util import view_as_windows as viewW
import numpy as np

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