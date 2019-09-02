from skimage import feature
import numpy as np


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describeImg(self, image, patch_size):
        H, W, = image.shape
        Hist = []
        assert (H == W)
        N = H // patch_size
        for i in range(N):
            for j in range(N):
                patch = image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                hist = self.describe(patch)
                Hist.append(hist)

        feat = np.zeros(1)
        #         print(len(Hist))
        for hist in Hist:
            feat = np.concatenate((feat, hist), axis=0)

        # print(feat)
        return feat

    def describe(self, image, eps=1e-7):#describe path
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist



#sample
if __name__ == '__main__':
    descLBP = LocalBinaryPatterns(9, 3)
