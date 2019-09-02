import numpy as np
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel


def get_gabor_kernels(num_theta=8,frequency_set = [1./1., 1./2.,1./3.,1./4.,1./5.] , sigma_set = [2*np.pi]):
    kernels = []
    for theta in range(num_theta):
        theta = float(theta) / num_theta * np.pi
        for sigma in sigma_set:
            for frequency in frequency_set:
                kernel = gabor_kernel(frequency, theta=theta,
                                      sigma_x=sigma, sigma_y=sigma)
                kernels.append(kernel)
    return kernels


class getTexture_Gabor:
    def __init__(self, num_theta=8,frequency_set = [1./1., 1./2.,1./3.,1./4.,1./5.] , sigma_set = [2*np.pi]):
        self.kernels = get_gabor_kernels(num_theta, frequency_set , sigma_set)
        self.num_theta = num_theta
        self.frequency_set = frequency_set
        self.sigma_set = sigma_set

    def compute_feats(self, image):

        mag_feat = []
        for k, kernel in enumerate(self.kernels):
            filtered_real = ndi.convolve(image, np.real(kernel), mode='wrap')
            filtered_imag = ndi.convolve(image, np.imag(kernel), mode='wrap')

            filtered_mag = np.sqrt(filtered_real * filtered_real + filtered_imag * filtered_imag)
            mag_feat.append(filtered_mag.reshape(-1))

        mag_feats = np.concatenate(mag_feat, axis=0)
        #         print(np.linalg.norm(mag_feats))
        mag_feats /= np.linalg.norm(mag_feats) + 1e-7
        #         print(mag_feats.shape)


        return mag_feats

    def describeImg(self, image, patch_size):
        H, W, = image.shape
        Hist = []
        assert (H == W)
        N = H // patch_size
        for i in range(N):
            for j in range(N):
                patch = image[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                hist = self.compute_feats(patch)
                Hist.append(hist)

        feat = np.zeros(1)

        for hist in Hist:
            feat = np.concatenate((feat, hist), axis=0)

#sample
if __name__ == '__main__':
    descGabor = getTexture_Gabor()